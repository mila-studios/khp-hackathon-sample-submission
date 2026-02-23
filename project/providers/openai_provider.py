"""OpenAI provider implementation"""

from typing import List, Optional
import asyncio
from openai import AsyncOpenAI, OpenAI

from .base import BaseLLMProvider, LLMMessage, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        super().__init__(api_key, model, temperature, max_tokens, **kwargs)
        self.system_prompt = system_prompt or ""
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        
        base = [{"role": msg.role, "content": msg.content} for msg in messages]
        if self.system_prompt and (not base or base[0].get("role") != "system"):
            openai_messages = [{"role": "system", "content": self.system_prompt}] + base
        else:
            openai_messages = base
        
        # Build parameters, excluding temperature if it's None or not supported
        params = {
            "model": self.model,
            "messages": openai_messages,
            "max_completion_tokens": self._get_max_tokens(max_tokens),
        }
        
        # Only add temperature if it's explicitly set and not None
        temp_value = self._get_temperature(temperature)
        if temp_value is not None:
            params["temperature"] = temp_value
        
        response = await self.async_client.chat.completions.create(
            **params,
            **{**self.extra_params, **kwargs}
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            metadata={"response_id": response.id}
        )
    
    def generate_sync(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Synchronous version using sync client"""
        
        base = [{"role": msg.role, "content": msg.content} for msg in messages]
        if self.system_prompt and (not base or base[0].get("role") != "system"):
            openai_messages = [{"role": "system", "content": self.system_prompt}] + base
        else:
            openai_messages = base
        
        # Build parameters, excluding temperature if it's None or not supported
        params = {
            "model": self.model,
            "messages": openai_messages,
            "max_completion_tokens": self._get_max_tokens(max_tokens),
        }
        
        # Only add temperature if it's explicitly set and not None
        temp_value = self._get_temperature(temperature)
        if temp_value is not None:
            params["temperature"] = temp_value
        
        response = self.sync_client.chat.completions.create(
            **params,
            **{**self.extra_params, **kwargs}
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            metadata={"response_id": response.id}
        )

