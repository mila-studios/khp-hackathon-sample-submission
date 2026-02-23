"""LLM-as-Judge guardrail implementation"""

import json
import time
from typing import Optional, Dict, Any

from .base import BaseGuardrail, GuardrailResult, GuardrailConfig, GuardrailStatus, EvaluationType
from providers.base import BaseLLMProvider, LLMMessage
from ..prompt_templates.guardrail_prompt_template import DEFAULT_USER_INPUT_PROMPT, DEFAULT_LLM_OUTPUT_PROMPT

class LLMJudgeGuardrail(BaseGuardrail):
    """
    LLM-as-Judge guardrail that uses an LLM to evaluate content
    
    Supports two evaluation types:
    1. USER_INPUT: Evaluates user queries/prompts before processing
    2. LLM_OUTPUT: Evaluates LLM responses before returning to user
    
    Each type uses a specialized system prompt for optimal evaluation.
    """
    
    def __init__(
        self,
        config: GuardrailConfig,
        llm_provider: BaseLLMProvider,
        user_input_prompt: Optional[str] = None,
        llm_output_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_format: str = "json"
    ):
        """
        Initialize LLM Judge guardrail with support for both input and output evaluation
        
        Args:
            config: Guardrail configuration
            llm_provider: LLM provider for evaluations
            user_input_prompt: Custom system prompt for evaluating user inputs (optional)
            llm_output_prompt: Custom system prompt for evaluating LLM outputs (optional)
            system_prompt: Unified system prompt for both input and output (backward compatibility)
            response_format: Expected response format (default: "json")
        """
        super().__init__(config)
        self.llm_provider = llm_provider
        
        # Support both separate prompts and unified system_prompt (backward compatibility)
        if system_prompt:
            self.user_input_prompt = system_prompt
            self.llm_output_prompt = system_prompt
        else:
            self.user_input_prompt = user_input_prompt or DEFAULT_USER_INPUT_PROMPT
            self.llm_output_prompt = llm_output_prompt or DEFAULT_LLM_OUTPUT_PROMPT
        
        self.response_format = response_format
    
    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        """
        Evaluate content using LLM as judge.

        Args:
            content: The content to evaluate
            context: Optional context information
            evaluation_type: USER_INPUT or LLM_OUTPUT

        Returns:
            GuardrailResult with evaluation details
        """
        start_time = time.time()
        try:
            system_prompt = self._get_system_prompt(evaluation_type)
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=self._format_evaluation_prompt(content, context, evaluation_type),
                ),
            ]
            response = self._generate_with_retry(messages)
            
            # Parse response
            evaluation = self._parse_llm_response(response.content)
            
            # Determine status based on evaluation
            status = self._determine_status(evaluation)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            return GuardrailResult(
                status=status,
                score=evaluation.get("score"),
                reasoning=evaluation.get("reasoning"),
                metadata={
                    "evaluation_type": evaluation_type.value,
                    "violations": evaluation.get("violations", []),
                    "risk_level": evaluation.get("risk_level"),
                    "indicators": evaluation.get("indicators", []),
                    "llm_response": response.content,
                    "model": response.model,
                    "usage": response.usage,
                },
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if self.config.fail_open:
                status = GuardrailStatus.PASS
            else:
                status = GuardrailStatus.ERROR
            is_parse_error = isinstance(e, ValueError) and "parse" in str(e).lower()
            return GuardrailResult(
                status=status,
                reasoning=f"Error during evaluation: {str(e)}",
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "is_parse_error": is_parse_error,
                    "evaluation_type": evaluation_type.value,
                    "fail_open": self.config.fail_open,
                },
                latency_ms=latency_ms,
            )

    def _get_system_prompt(self, evaluation_type: EvaluationType) -> str:
        """Get the appropriate system prompt based on evaluation type"""
        if evaluation_type == EvaluationType.USER_INPUT:
            return self.user_input_prompt
        elif evaluation_type == EvaluationType.LLM_OUTPUT:
            return self.llm_output_prompt
        else:
            # Fallback to user input prompt
            return self.user_input_prompt
    
    def _format_evaluation_prompt(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]],
        evaluation_type: EvaluationType
    ) -> str:
        """Format the evaluation prompt based on evaluation type"""
        if evaluation_type == EvaluationType.USER_INPUT:
            prompt = f"Evaluate the following user input:\n\n{content}"
        else:
            prompt = f"Evaluate the following LLM output:\n\n{content}"
        
        if context:
            prompt += f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        prompt += "\n\nProvide your evaluation in JSON format."
        
        return prompt
    
    def _generate_with_retry(self, messages):
        """Generate with retry logic (sync)."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return self.llm_provider.generate_sync(messages)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
        raise last_error
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate the LLM response
        
        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
            else:
                # Fallback: try to parse the whole response
                parsed = json.loads(response_text)
            
            # Validate that we have the required keys
            return self._validate_and_normalize_response(parsed)
                
        except json.JSONDecodeError as e:
            # Raise exception to be handled by outer error handling
            # This will respect the fail_open setting
            raise ValueError(f"Unable to parse LLM response as JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing LLM response: {str(e)}")
    
    def _validate_and_normalize_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the parsed response to ensure expected keys exist
        
        Args:
            parsed: The parsed JSON response
            
        Returns:
            Normalized response with all expected keys
            
        Raises:
            ValueError: If critical keys are missing or invalid
        """
        # Resolve harmful: accept "harmful" or "safe" (safe = not harmful)
        if "harmful" in parsed:
            harmful = bool(parsed["harmful"])
        elif "safe" in parsed:
            harmful = not bool(parsed["safe"])
        else:
            raise ValueError("Missing required key: need 'harmful' or 'safe' in LLM response")
        
        if "score" not in parsed:
            raise ValueError("Missing required key 'score' in LLM response")
        
        # Validate types (only if present)
        if "harmful" in parsed and not isinstance(parsed["harmful"], bool):
            raise ValueError(f"Invalid type for 'harmful': expected bool, got {type(parsed['harmful']).__name__}")
        if "safe" in parsed and not isinstance(parsed["safe"], bool):
            raise ValueError(f"Invalid type for 'safe': expected bool, got {type(parsed['safe']).__name__}")
        
        if not isinstance(parsed["score"], (int, float)):
            raise ValueError(f"Invalid type for 'score': expected number, got {type(parsed['score']).__name__}")
        
        # Validate score range
        score = float(parsed["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score {score} is out of valid range [0.0, 1.0]")
        
        # Normalize response with defaults for optional keys
        normalized = {
            "harmful": harmful,
            "safe": not harmful,
            "score": score,
            "reasoning": parsed.get("reasoning", "No reasoning provided"),
            "violations": parsed.get("violations", []),
            "risk_level": parsed.get("risk_level", "unknown"),
            "indicators": parsed.get("indicators", [])
        }
        
        return normalized
    
    def _determine_status(self, evaluation: Dict[str, Any]) -> GuardrailStatus:
        """Determine guardrail status from evaluation"""
        score = evaluation.get("score", 0.5)
        is_harmful = evaluation.get("harmful", False)
        
        # If explicitly marked as harmful
        if is_harmful:
            return GuardrailStatus.FAIL
        
        # Use threshold to determine status (higher score = more dangerous)
        if score >= self.config.threshold:
            return GuardrailStatus.FAIL
        else:
            return GuardrailStatus.PASS

