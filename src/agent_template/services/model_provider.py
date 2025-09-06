"""
Multi-model support abstraction layer.

This module provides a unified interface for different AI model providers
including OpenAI, Anthropic, local models, and future providers.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Callable

import httpx
import structlog

from ..config import settings
from ..models.messages import Message, MessageRole


logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Supported model types."""
    
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"


class ProviderType(str, Enum):
    """AI model providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE = "azure"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OLLAMA = "ollama"


@dataclass
class ModelInfo:
    """Model information and capabilities."""
    
    id: str
    name: str
    provider: ProviderType
    model_type: ModelType
    max_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    cost_per_token: Optional[float] = None
    context_window: int = 4096
    
    def __post_init__(self):
        """Validate model info after creation."""
        if self.max_tokens > self.context_window:
            self.max_tokens = self.context_window


@dataclass  
class ModelRequest:
    """Request configuration for model calls."""
    
    messages: List[Message]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    # Provider-specific options
    provider_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.provider_options is None:
            self.provider_options = {}


@dataclass
class ModelResponse:
    """Response from model provider."""
    
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    tool_calls: List[Dict[str, Any]] = None
    
    # Metadata
    response_time: float = 0.0
    provider: Optional[str] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class ModelProvider(ABC):
    """Abstract base class for AI model providers."""
    
    def __init__(self, provider_type: ProviderType):
        self.provider_type = provider_type
        self.client: Optional[Any] = None
        self._models: Dict[str, ModelInfo] = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def chat_completion(
        self,
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion."""
        pass
    
    @abstractmethod  
    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        pass
    
    async def validate_request(self, request: ModelRequest) -> bool:
        """Validate a model request."""
        model_info = await self.get_model_info(request.model)
        if not model_info:
            return False
        
        # Check token limits
        total_tokens = sum(msg.estimate_tokens() for msg in request.messages)
        if request.max_tokens:
            total_tokens += request.max_tokens
        
        return total_tokens <= model_info.context_window
    
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        if hasattr(self.client, 'close'):
            await self.client.close()


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""
    
    def __init__(self):
        super().__init__(ProviderType.OPENAI)
        self.api_key = settings.models.openai_api_key
        self.base_url = "https://api.openai.com/v1"
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Load model information
            await self._load_models()
            
            logger.info("OpenAI provider initialized")
            
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
    
    async def chat_completion(
        self, 
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using OpenAI."""
        if not await self.validate_request(request):
            raise ValueError("Invalid request for model")
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(request.messages)
        
        kwargs = {
            "model": request.model,
            "messages": openai_messages,
            "temperature": request.temperature,
            "stream": request.stream,
            **request.provider_options
        }
        
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        
        if request.tools:
            kwargs["tools"] = request.tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._stream_completion(kwargs, start_time)
            else:
                response = await self.client.chat.completions.create(**kwargs)
                return self._parse_response(response, start_time)
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _stream_completion(
        self, 
        kwargs: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion from OpenAI."""
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def _parse_response(self, response: Any, start_time: float) -> ModelResponse:
        """Parse OpenAI response."""
        choice = response.choices[0]
        
        # Extract tool calls if present
        tool_calls = []
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            response_time=time.time() - start_time,
            provider="openai",
            request_id=getattr(response, 'id', None)
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_msg = {
                "role": message.role.value,
                "content": message.get_text_content()
            }
            
            # Add tool calls if present
            if message.tool_calls:
                openai_msg["tool_calls"] = message.tool_calls
            
            # Add tool call ID if present
            if message.tool_call_id:
                openai_msg["tool_call_id"] = message.tool_call_id
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    async def _load_models(self) -> None:
        """Load available OpenAI models."""
        # Pre-defined model information (could be fetched from API)
        models = {
            "gpt-4-turbo-preview": ModelInfo(
                id="gpt-4-turbo-preview",
                name="GPT-4 Turbo Preview",
                provider=ProviderType.OPENAI,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=128000,
                supports_tools=True,
                supports_vision=True
            ),
            "gpt-4": ModelInfo(
                id="gpt-4",
                name="GPT-4",
                provider=ProviderType.OPENAI,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=8192,
                supports_tools=True
            ),
            "gpt-3.5-turbo": ModelInfo(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider=ProviderType.OPENAI,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=16384,
                supports_tools=True
            )
        }
        
        self._models.update(models)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        return list(self._models.values())
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get OpenAI model information."""
        return self._models.get(model_id)


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""
    
    def __init__(self):
        super().__init__(ProviderType.ANTHROPIC)
        self.api_key = settings.models.anthropic_api_key
    
    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            await self._load_models()
            logger.info("Anthropic provider initialized")
            
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
    
    async def chat_completion(
        self,
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using Anthropic."""
        if not await self.validate_request(request):
            raise ValueError("Invalid request for model")
        
        # Convert messages to Anthropic format
        system_message, messages = self._convert_messages(request.messages)
        
        kwargs = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": request.stream,
            **request.provider_options
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        else:
            kwargs["max_tokens"] = 1024  # Anthropic requires max_tokens
        
        if request.tools:
            kwargs["tools"] = self._convert_tools(request.tools)
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._stream_completion(kwargs, start_time)
            else:
                response = await self.client.messages.create(**kwargs)
                return self._parse_response(response, start_time)
                
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _stream_completion(
        self,
        kwargs: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Anthropic."""
        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    def _parse_response(self, response: Any, start_time: float) -> ModelResponse:
        """Parse Anthropic response."""
        # Extract text content
        content = ""
        tool_calls = []
        
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append({
                    "id": content_block.id,
                    "type": "function",
                    "function": {
                        "name": content_block.name,
                        "arguments": json.dumps(content_block.input)
                    }
                })
        
        return ModelResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason or "stop",
            tool_calls=tool_calls,
            response_time=time.time() - start_time,
            provider="anthropic",
            request_id=response.id
        )
    
    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert internal messages to Anthropic format."""
        system_message = None
        anthropic_messages = []
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                system_message = message.get_text_content()
                continue
            
            # Map roles
            role = "user" if message.role == MessageRole.USER else "assistant"
            
            anthropic_msg = {
                "role": role,
                "content": message.get_text_content()
            }
            
            # Handle tool results
            if message.role == MessageRole.TOOL:
                anthropic_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.get_text_content()
                    }]
                }
            
            anthropic_messages.append(anthropic_msg)
        
        return system_message, anthropic_messages
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        
        return anthropic_tools
    
    async def _load_models(self) -> None:
        """Load available Anthropic models."""
        models = {
            "claude-3-sonnet-20240229": ModelInfo(
                id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                provider=ProviderType.ANTHROPIC,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=200000,
                supports_tools=True,
                supports_vision=True
            ),
            "claude-3-haiku-20240307": ModelInfo(
                id="claude-3-haiku-20240307", 
                name="Claude 3 Haiku",
                provider=ProviderType.ANTHROPIC,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=200000,
                supports_tools=True,
                supports_vision=True
            )
        }
        
        self._models.update(models)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Anthropic models."""
        return list(self._models.values())
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get Anthropic model information."""
        return self._models.get(model_id)


class LocalProvider(ModelProvider):
    """Local model provider (e.g., Ollama, vLLM)."""
    
    def __init__(self):
        super().__init__(ProviderType.LOCAL)
        self.base_url = settings.models.local_model_url or "http://localhost:11434"
        self.model_name = settings.models.local_model_name or "llama2"
    
    async def initialize(self) -> None:
        """Initialize local model client."""
        self.client = httpx.AsyncClient(timeout=60.0)
        
        try:
            # Test connection
            await self._test_connection()
            await self._load_models()
            
            logger.info("Local provider initialized", url=self.base_url)
            
        except Exception as e:
            logger.error(f"Failed to initialize local provider: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """Test connection to local model server."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to local model server: {e}")
    
    async def chat_completion(
        self,
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using local model."""
        # Convert messages to local format (usually just the last message)
        prompt = self._convert_messages_to_prompt(request.messages)
        
        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                **(request.provider_options or {})
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._stream_completion(payload, start_time)
            else:
                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                return self._parse_response(response.json(), start_time)
                
        except Exception as e:
            logger.error(f"Local model API error: {e}")
            raise
    
    async def _stream_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion from local model."""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Local model streaming error: {e}")
            raise
    
    def _parse_response(self, response_data: Dict[str, Any], start_time: float) -> ModelResponse:
        """Parse local model response."""
        return ModelResponse(
            content=response_data.get("response", ""),
            model=response_data.get("model", self.model_name),
            usage={
                "prompt_tokens": 0,  # Local models might not provide this
                "completion_tokens": 0,
                "total_tokens": 0
            },
            finish_reason="stop",
            response_time=time.time() - start_time,
            provider="local"
        )
    
    def _convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a single prompt for local models."""
        prompt_parts = []
        
        for message in messages:
            role = message.role.value.title()
            content = message.get_text_content()
            prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def _load_models(self) -> None:
        """Load available local models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            models = {}
            
            for model_info in models_data.get("models", []):
                model_id = model_info.get("name", "")
                if model_id:
                    models[model_id] = ModelInfo(
                        id=model_id,
                        name=model_info.get("name", model_id),
                        provider=ProviderType.LOCAL,
                        model_type=ModelType.CHAT,
                        max_tokens=2048,  # Default
                        context_window=4096,  # Default
                        supports_streaming=True,
                        supports_tools=False  # Most local models don't support tools
                    )
            
            self._models.update(models)
            
        except Exception as e:
            logger.warning(f"Could not load local models: {e}")
            # Add default model
            self._models[self.model_name] = ModelInfo(
                id=self.model_name,
                name=self.model_name,
                provider=ProviderType.LOCAL,
                model_type=ModelType.CHAT,
                max_tokens=2048,
                context_window=4096,
                supports_streaming=True
            )
    
    async def list_models(self) -> List[ModelInfo]:
        """List available local models."""
        return list(self._models.values())
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get local model information."""
        return self._models.get(model_id)


class DeepSeekProvider(ModelProvider):
    """DeepSeek API provider (OpenAI-compatible)."""
    
    def __init__(self):
        super().__init__(ProviderType.DEEPSEEK)
        self.api_key = settings.models.deepseek_api_key
        self.base_url = settings.models.deepseek_base_url
    
    async def initialize(self) -> None:
        """Initialize DeepSeek client."""
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")

        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            await self._load_models()
            logger.info("DeepSeek provider initialized")

        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            raise RuntimeError(f"DeepSeek initialization failed: {e}")
    
    async def chat_completion(
        self, 
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using DeepSeek (OpenAI-compatible API)."""
        if not await self.validate_request(request):
            raise ValueError("Invalid request for model")
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(request.messages)
        
        kwargs = {
            "model": request.model,
            "messages": openai_messages,
            "temperature": request.temperature,
            "stream": request.stream,
            **request.provider_options
        }
        
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._stream_completion(kwargs, start_time)
            else:
                response = await self.client.chat.completions.create(**kwargs)
                return self._parse_response(response, start_time)
                
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    async def _stream_completion(
        self, 
        kwargs: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion from DeepSeek."""
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"DeepSeek streaming error: {e}")
            raise
    
    def _parse_response(self, response: Any, start_time: float) -> ModelResponse:
        """Parse DeepSeek response."""
        choice = response.choices[0]
        
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=choice.finish_reason,
            response_time=time.time() - start_time,
            provider="deepseek"
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []

        for message in messages:
            # Handle both enum and string roles
            role = message.role.value if hasattr(message.role, 'value') else message.role
            openai_msg = {
                "role": role,
                "content": message.get_text_content()
            }
            openai_messages.append(openai_msg)

        return openai_messages
    
    async def _load_models(self) -> None:
        """Load available DeepSeek models."""
        models = {
            "deepseek-chat": ModelInfo(
                id="deepseek-chat",
                name="DeepSeek Chat",
                provider=ProviderType.DEEPSEEK,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=32768,
                supports_tools=False
            ),
            "deepseek-coder": ModelInfo(
                id="deepseek-coder",
                name="DeepSeek Coder",
                provider=ProviderType.DEEPSEEK,
                model_type=ModelType.CHAT,
                max_tokens=4096,
                context_window=16384,
                supports_tools=False
            ),
        }
        
        self._models.update(models)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available DeepSeek models."""
        return list(self._models.values())
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get DeepSeek model information."""
        return self._models.get(model_id)


class QwenProvider(ModelProvider):
    """Qwen API provider (OpenAI-compatible)."""
    
    def __init__(self):
        super().__init__(ProviderType.QWEN)
        self.api_key = settings.models.qwen_api_key
        self.base_url = settings.models.qwen_base_url
    
    async def initialize(self) -> None:
        """Initialize Qwen client."""
        if not self.api_key:
            raise ValueError("Qwen API key not configured")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            await self._load_models()
            logger.info("Qwen provider initialized")
            
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
    
    async def chat_completion(
        self, 
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using Qwen (OpenAI-compatible API)."""
        if not await self.validate_request(request):
            raise ValueError("Invalid request for model")
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(request.messages)
        
        kwargs = {
            "model": request.model,
            "messages": openai_messages,
            "temperature": request.temperature,
            "stream": request.stream,
            **request.provider_options
        }
        
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._stream_completion(kwargs, start_time)
            else:
                response = await self.client.chat.completions.create(**kwargs)
                return self._parse_response(response, start_time)
                
        except Exception as e:
            logger.error(f"Qwen API error: {e}")
            raise
    
    async def _stream_completion(
        self, 
        kwargs: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Qwen."""
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Qwen streaming error: {e}")
            raise
    
    def _parse_response(self, response: Any, start_time: float) -> ModelResponse:
        """Parse Qwen response."""
        choice = response.choices[0]
        
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=choice.finish_reason,
            response_time=time.time() - start_time,
            provider="qwen"
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_msg = {
                "role": message.role.value,
                "content": message.get_text_content()
            }
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    async def _load_models(self) -> None:
        """Load available Qwen models."""
        models = {
            "qwen-turbo": ModelInfo(
                id="qwen-turbo",
                name="Qwen Turbo",
                provider=ProviderType.QWEN,
                model_type=ModelType.CHAT,
                max_tokens=2048,
                context_window=8192,
                supports_tools=True
            ),
            "qwen-plus": ModelInfo(
                id="qwen-plus",
                name="Qwen Plus",
                provider=ProviderType.QWEN,
                model_type=ModelType.CHAT,
                max_tokens=2048,
                context_window=32768,
                supports_tools=True
            ),
            "qwen-max": ModelInfo(
                id="qwen-max",
                name="Qwen Max",
                provider=ProviderType.QWEN,
                model_type=ModelType.CHAT,
                max_tokens=2048,
                context_window=32768,
                supports_tools=True
            ),
        }
        
        self._models.update(models)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Qwen models."""
        return list(self._models.values())
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get Qwen model information."""
        return self._models.get(model_id)


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""
    
    def __init__(self):
        super().__init__(ProviderType.OLLAMA)
        self.base_url = settings.models.ollama_base_url
        self.default_model = settings.models.ollama_model
    
    async def initialize(self) -> None:
        """Initialize Ollama client."""
        self.client = httpx.AsyncClient(timeout=60.0)
        
        try:
            # Test connection
            await self._test_connection()
            await self._load_models()
            
            logger.info("Ollama provider initialized", url=self.base_url)
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama server: {e}")
    
    async def chat_completion(
        self,
        request: ModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using Ollama."""
        # Convert messages to Ollama format
        messages = self._convert_messages(request.messages)
        
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                **(request.provider_options or {})
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._stream_completion(payload, start_time)
            else:
                response = await self.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                return self._parse_response(response.json(), start_time)
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def _stream_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama."""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    def _parse_response(self, response_data: Dict[str, Any], start_time: float) -> ModelResponse:
        """Parse Ollama response."""
        message = response_data.get("message", {})
        content = message.get("content", "")
        
        return ModelResponse(
            content=content,
            model=response_data.get("model", self.default_model),
            usage={
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
            },
            finish_reason="stop",
            response_time=time.time() - start_time,
            provider="ollama"
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to Ollama format."""
        ollama_messages = []
        
        for message in messages:
            ollama_msg = {
                "role": message.role.value,
                "content": message.get_text_content()
            }
            ollama_messages.append(ollama_msg)
        
        return ollama_messages
    
    async def _load_models(self) -> None:
        """Load available Ollama models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            models = {}
            
            for model_info in models_data.get("models", []):
                model_name = model_info.get("name", "")
                if model_name:
                    # Parse model name (remove tag if present)
                    base_name = model_name.split(":")[0]
                    
                    models[model_name] = ModelInfo(
                        id=model_name,
                        name=base_name.title(),
                        provider=ProviderType.OLLAMA,
                        model_type=ModelType.CHAT,
                        max_tokens=2048,
                        context_window=4096,  # Default, varies by model
                        supports_streaming=True,
                        supports_tools=False
                    )
            
            self._models.update(models)
            logger.info(f"Loaded {len(models)} Ollama models")
            
        except Exception as e:
            logger.warning(f"Could not load Ollama models: {e}")
            # Add default model
            self._models[self.default_model] = ModelInfo(
                id=self.default_model,
                name=self.default_model.title(),
                provider=ProviderType.OLLAMA,
                model_type=ModelType.CHAT,
                max_tokens=2048,
                context_window=4096,
                supports_streaming=True
            )
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Ollama models."""
        return list(self._models.values())
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get Ollama model information."""
        return self._models.get(model_id)


class ModelManager:
    """
    Manager for multiple model providers with routing and fallbacks.
    """
    
    def __init__(self):
        self._providers: Dict[ProviderType, ModelProvider] = {}
        self._initialized = False
        self._default_provider: Optional[ProviderType] = None
    
    async def initialize(self) -> None:
        """Initialize all configured providers."""
        if self._initialized:
            return
        
        # Initialize providers based on configuration
        if settings.models.openai_api_key:
            self._providers[ProviderType.OPENAI] = OpenAIProvider()
        
        if settings.models.anthropic_api_key:
            self._providers[ProviderType.ANTHROPIC] = AnthropicProvider()
        
        if settings.models.local_model_url:
            self._providers[ProviderType.LOCAL] = LocalProvider()
        
        if settings.models.deepseek_api_key:
            self._providers[ProviderType.DEEPSEEK] = DeepSeekProvider()
        
        if settings.models.qwen_api_key:
            self._providers[ProviderType.QWEN] = QwenProvider()
        
        # Ollama is always available if the server is running
        try:
            self._providers[ProviderType.OLLAMA] = OllamaProvider()
        except Exception:
            # Ollama server not available, skip
            pass
        
        # Initialize all providers
        for provider in self._providers.values():
            try:
                await provider.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider.provider_type}: {e}")
        
        # Set default provider
        default_provider_name = settings.models.default_provider
        if default_provider_name in [p.value for p in ProviderType]:
            self._default_provider = ProviderType(default_provider_name)
        elif self._providers:
            self._default_provider = next(iter(self._providers.keys()))
        
        self._initialized = True
        logger.info("ModelManager initialized", 
                   providers=list(self._providers.keys()),
                   default=self._default_provider)
    
    async def chat_completion(
        self,
        request: ModelRequest,
        provider: Optional[ProviderType] = None
    ) -> Union[ModelResponse, AsyncGenerator[str, None]]:
        """Generate chat completion using specified or default provider."""
        if not self._initialized:
            await self.initialize()
        
        # Determine provider
        target_provider = provider or self._default_provider
        if target_provider not in self._providers:
            raise ValueError(f"Provider {target_provider} not available")
        
        provider_instance = self._providers[target_provider]
        return await provider_instance.chat_completion(request)
    
    async def list_all_models(self) -> Dict[ProviderType, List[ModelInfo]]:
        """List models from all providers."""
        if not self._initialized:
            await self.initialize()
        
        all_models = {}
        for provider_type, provider in self._providers.items():
            try:
                models = await provider.list_models()
                all_models[provider_type] = models
            except Exception as e:
                logger.error(f"Failed to list models for {provider_type}: {e}")
                all_models[provider_type] = []
        
        return all_models
    
    async def get_model_info(
        self,
        model_id: str,
        provider: Optional[ProviderType] = None
    ) -> Optional[ModelInfo]:
        """Get model information."""
        if not self._initialized:
            await self.initialize()
        
        if provider:
            if provider in self._providers:
                return await self._providers[provider].get_model_info(model_id)
            return None
        
        # Search all providers
        for provider_instance in self._providers.values():
            model_info = await provider_instance.get_model_info(model_id)
            if model_info:
                return model_info
        
        return None
    
    async def shutdown(self) -> None:
        """Shutdown all providers."""
        for provider in self._providers.values():
            try:
                await provider.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down provider: {e}")
        
        self._providers.clear()
        self._initialized = False
        logger.info("ModelManager shut down")
    
    @property
    def available_providers(self) -> List[ProviderType]:
        """Get list of available providers."""
        return list(self._providers.keys())
    
    @property
    def default_provider(self) -> Optional[ProviderType]:
        """Get default provider."""
        return self._default_provider


# Global model manager instance
model_manager = ModelManager()