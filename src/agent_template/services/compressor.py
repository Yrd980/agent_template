"""
Message compression and context optimization service.

This module provides intelligent message compression, context optimization,
and history summarization to manage token limits efficiently.
"""

import asyncio
import hashlib
import json
import lz4.frame
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque

import structlog
import xxhash

from ..config import settings
from ..models.messages import Message, MessageHistory, MessageRole
from ..services.model_provider import ModelManager, ModelRequest, ProviderType


logger = structlog.get_logger(__name__)


class CompressionMethod(str, Enum):
    """Available compression methods."""
    
    LZ4 = "lz4"
    SUMMARIZATION = "summarization"
    KEY_EXTRACTION = "key_extraction"
    SEMANTIC_DEDUPLICATION = "semantic_deduplication"
    HYBRID = "hybrid"


class CompressionLevel(int, Enum):
    """Compression levels."""
    
    LIGHT = 1    # Basic compression, preserve most details
    MEDIUM = 2   # Balanced compression
    HEAVY = 3    # Aggressive compression, preserve key points only
    EXTREME = 4  # Maximum compression, essential information only


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    
    compressed_content: Union[str, bytes]
    original_size: int
    compressed_size: int
    compression_ratio: float
    method: CompressionMethod
    level: CompressionLevel
    metadata: Dict[str, Any]
    processing_time: float
    
    def __post_init__(self):
        """Calculate compression ratio if not provided."""
        if self.compression_ratio == 0 and self.original_size > 0:
            self.compression_ratio = self.compressed_size / self.original_size


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""
    
    def __init__(self, method: CompressionMethod):
        self.method = method
        self.stats = {
            "compressions": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "total_time": 0.0,
            "average_ratio": 0.0
        }
    
    @abstractmethod
    async def compress(
        self,
        content: Union[str, List[Message]],
        level: CompressionLevel = CompressionLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """Compress content using this strategy."""
        pass
    
    @abstractmethod
    async def decompress(
        self,
        compressed_data: Union[str, bytes],
        metadata: Dict[str, Any]
    ) -> str:
        """Decompress content (if possible)."""
        pass
    
    def update_stats(self, result: CompressionResult) -> None:
        """Update compression statistics."""
        self.stats["compressions"] += 1
        self.stats["total_original_size"] += result.original_size
        self.stats["total_compressed_size"] += result.compressed_size
        self.stats["total_time"] += result.processing_time
        
        if self.stats["compressions"] > 0:
            self.stats["average_ratio"] = (
                self.stats["total_compressed_size"] / 
                self.stats["total_original_size"]
            )


class LZ4CompressionStrategy(CompressionStrategy):
    """LZ4-based compression for raw text."""
    
    def __init__(self):
        super().__init__(CompressionMethod.LZ4)
    
    async def compress(
        self,
        content: Union[str, List[Message]],
        level: CompressionLevel = CompressionLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """Compress using LZ4."""
        start_time = time.time()
        
        # Convert to text if messages
        if isinstance(content, list):
            text = self._messages_to_text(content)
        else:
            text = content
        
        original_bytes = text.encode('utf-8')
        original_size = len(original_bytes)
        
        # Apply LZ4 compression with different levels
        compression_level = min(int(level), 12)  # LZ4 max level is 12
        
        try:
            compressed_bytes = lz4.frame.compress(
                original_bytes,
                compression_level=compression_level,
                auto_flush=True
            )
            
            compressed_size = len(compressed_bytes)
            processing_time = time.time() - start_time
            
            result = CompressionResult(
                compressed_content=compressed_bytes,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compressed_size / original_size,
                method=self.method,
                level=level,
                metadata={"encoding": "utf-8", "lz4_level": compression_level},
                processing_time=processing_time
            )
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"LZ4 compression failed: {e}")
            raise
    
    async def decompress(
        self,
        compressed_data: bytes,
        metadata: Dict[str, Any]
    ) -> str:
        """Decompress LZ4 data."""
        try:
            decompressed_bytes = lz4.frame.decompress(compressed_data)
            encoding = metadata.get("encoding", "utf-8")
            return decompressed_bytes.decode(encoding)
            
        except Exception as e:
            logger.error(f"LZ4 decompression failed: {e}")
            raise
    
    def _messages_to_text(self, messages: List[Message]) -> str:
        """Convert messages to text format."""
        lines = []
        for msg in messages:
            role = msg.role.value.upper()
            content = msg.get_text_content()
            timestamp = msg.timestamp.isoformat()
            lines.append(f"[{timestamp}] {role}: {content}")
        
        return '\n'.join(lines)


class SummarizationStrategy(CompressionStrategy):
    """AI-based summarization for semantic compression."""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__(CompressionMethod.SUMMARIZATION)
        self.model_manager = model_manager
        self._summary_cache: Dict[str, str] = {}
    
    async def compress(
        self,
        content: Union[str, List[Message]],
        level: CompressionLevel = CompressionLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """Compress using AI summarization."""
        start_time = time.time()
        
        # Convert to text
        if isinstance(content, list):
            text = self._format_messages_for_summary(content)
            message_count = len(content)
        else:
            text = content
            message_count = 1
        
        original_size = len(text.encode('utf-8'))
        
        # Check cache first
        content_hash = self._hash_content(text)
        if content_hash in self._summary_cache:
            summary = self._summary_cache[content_hash]
        else:
            # Generate summary using AI
            summary = await self._generate_summary(text, level, message_count)
            self._summary_cache[content_hash] = summary
        
        compressed_size = len(summary.encode('utf-8'))
        processing_time = time.time() - start_time
        
        result = CompressionResult(
            compressed_content=summary,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            method=self.method,
            level=level,
            metadata={
                "content_hash": content_hash,
                "message_count": message_count,
                "summary_type": self._get_summary_type(level)
            },
            processing_time=processing_time
        )
        
        self.update_stats(result)
        return result
    
    async def decompress(
        self,
        compressed_data: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Return the summary (can't fully decompress summarization)."""
        return compressed_data
    
    async def _generate_summary(
        self,
        text: str,
        level: CompressionLevel,
        message_count: int
    ) -> str:
        """Generate AI summary based on compression level."""
        # Create summarization prompt based on level
        if level == CompressionLevel.LIGHT:
            instruction = "Provide a detailed summary preserving most key information and context."
            max_tokens = min(1000, len(text.split()) // 2)
        elif level == CompressionLevel.MEDIUM:
            instruction = "Provide a balanced summary capturing the main points and important details."
            max_tokens = min(500, len(text.split()) // 3)
        elif level == CompressionLevel.HEAVY:
            instruction = "Provide a concise summary focusing on the most important information."
            max_tokens = min(250, len(text.split()) // 4)
        else:  # EXTREME
            instruction = "Provide a very brief summary with only the essential information."
            max_tokens = min(100, len(text.split()) // 6)
        
        prompt = f"""Please summarize the following conversation or text:

{instruction}

The original text contains {message_count} message(s).

Text to summarize:
{text}

Summary:"""
        
        # Create summarization request
        from ..models.messages import Message as MessageModel
        
        summary_message = MessageModel(
            role=MessageRole.USER,
            content=prompt,
            session_id="compression_session"
        )
        
        request = ModelRequest(
            messages=[summary_message],
            model=settings.models.openai_model,  # Use configured model
            temperature=0.3,  # Lower temperature for consistent summaries
            max_tokens=max_tokens
        )
        
        try:
            response = await self.model_manager.chat_completion(request)
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                # Handle streaming response
                summary_parts = []
                async for chunk in response:
                    summary_parts.append(chunk)
                return ''.join(summary_parts).strip()
                
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple truncation
            return self._fallback_summary(text, max_tokens)
    
    def _fallback_summary(self, text: str, max_tokens: int) -> str:
        """Fallback summary when AI summarization fails."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        
        # Take first and last portions
        first_part = ' '.join(words[:max_tokens//2])
        last_part = ' '.join(words[-(max_tokens//2):])
        
        return f"{first_part}\n\n[... content truncated ...]\n\n{last_part}"
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """Format messages for summarization."""
        formatted_lines = []
        
        for i, msg in enumerate(messages):
            role = msg.role.value.title()
            content = msg.get_text_content()
            timestamp = msg.timestamp.strftime("%H:%M")
            
            formatted_lines.append(f"{i+1}. [{timestamp}] {role}: {content}")
        
        return '\n\n'.join(formatted_lines)
    
    def _get_summary_type(self, level: CompressionLevel) -> str:
        """Get summary type description."""
        types = {
            CompressionLevel.LIGHT: "detailed",
            CompressionLevel.MEDIUM: "balanced", 
            CompressionLevel.HEAVY: "concise",
            CompressionLevel.EXTREME: "brief"
        }
        return types.get(level, "balanced")
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content caching."""
        return xxhash.xxh64(content.encode('utf-8')).hexdigest()


class KeyExtractionStrategy(CompressionStrategy):
    """Extract key information and entities."""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__(CompressionMethod.KEY_EXTRACTION)
        self.model_manager = model_manager
    
    async def compress(
        self,
        content: Union[str, List[Message]],
        level: CompressionLevel = CompressionLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """Compress by extracting key information."""
        start_time = time.time()
        
        if isinstance(content, list):
            text = self._messages_to_structured_text(content)
        else:
            text = content
        
        original_size = len(text.encode('utf-8'))
        
        # Extract key information
        key_info = await self._extract_key_information(text, level)
        
        # Format as structured data
        compressed_content = json.dumps(key_info, ensure_ascii=False, indent=2)
        compressed_size = len(compressed_content.encode('utf-8'))
        processing_time = time.time() - start_time
        
        result = CompressionResult(
            compressed_content=compressed_content,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            method=self.method,
            level=level,
            metadata={"extraction_type": "structured", "format": "json"},
            processing_time=processing_time
        )
        
        self.update_stats(result)
        return result
    
    async def decompress(
        self,
        compressed_data: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Convert key information back to readable format."""
        try:
            key_info = json.loads(compressed_data)
            return self._format_key_info(key_info)
        except Exception as e:
            logger.error(f"Key extraction decompression failed: {e}")
            return compressed_data
    
    async def _extract_key_information(
        self,
        text: str,
        level: CompressionLevel
    ) -> Dict[str, Any]:
        """Extract key information using AI."""
        if level == CompressionLevel.LIGHT:
            instruction = "Extract detailed key information including context and nuances."
        elif level == CompressionLevel.MEDIUM:
            instruction = "Extract main topics, decisions, action items, and important facts."
        elif level == CompressionLevel.HEAVY:
            instruction = "Extract only the most critical information and key decisions."
        else:  # EXTREME
            instruction = "Extract only essential facts and final conclusions."
        
        prompt = f"""Analyze the following text and extract key information in JSON format.

{instruction}

Format the output as JSON with these categories:
- "topics": Main topics discussed
- "key_points": Important points or facts
- "decisions": Decisions made or conclusions reached
- "action_items": Tasks or actions mentioned
- "entities": Important people, places, or things mentioned
- "summary": Brief overview

Text to analyze:
{text}

Key Information (JSON):"""
        
        try:
            from ..models.messages import Message as MessageModel
            
            extraction_message = MessageModel(
                role=MessageRole.USER,
                content=prompt,
                session_id="extraction_session"
            )
            
            request = ModelRequest(
                messages=[extraction_message],
                model=settings.models.openai_model,
                temperature=0.1,  # Very low temperature for consistent extraction
                max_tokens=1000
            )
            
            response = await self.model_manager.chat_completion(request)
            
            if hasattr(response, 'content'):
                content = response.content.strip()
            else:
                # Handle streaming
                parts = []
                async for chunk in response:
                    parts.append(chunk)
                content = ''.join(parts).strip()
            
            # Try to parse as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to simple extraction
                return self._fallback_extraction(text)
                
        except Exception as e:
            logger.error(f"AI key extraction failed: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback key extraction when AI fails."""
        sentences = text.split('. ')
        words = text.split()
        
        return {
            "topics": ["General conversation"],
            "key_points": sentences[:3] if len(sentences) >= 3 else sentences,
            "decisions": [],
            "action_items": [],
            "entities": list(set(word for word in words if word.istitle()))[:10],
            "summary": sentences[0] if sentences else "No content available",
            "word_count": len(words),
            "extraction_method": "fallback"
        }
    
    def _messages_to_structured_text(self, messages: List[Message]) -> str:
        """Convert messages to structured text."""
        sections = []
        
        for msg in messages:
            role = msg.role.value.title()
            content = msg.get_text_content()
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            
            sections.append(f"=== {role} ({timestamp}) ===\n{content}")
        
        return '\n\n'.join(sections)
    
    def _format_key_info(self, key_info: Dict[str, Any]) -> str:
        """Format key information for display."""
        lines = ["=== KEY INFORMATION ===\n"]
        
        for category, items in key_info.items():
            if not items:
                continue
            
            lines.append(f"## {category.upper().replace('_', ' ')}")
            
            if isinstance(items, list):
                for item in items:
                    lines.append(f"- {item}")
            else:
                lines.append(f"{items}")
            
            lines.append("")
        
        return '\n'.join(lines)


class HybridCompressionStrategy(CompressionStrategy):
    """Combines multiple compression methods for optimal results."""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__(CompressionMethod.HYBRID)
        self.strategies = {
            CompressionMethod.LZ4: LZ4CompressionStrategy(),
            CompressionMethod.SUMMARIZATION: SummarizationStrategy(model_manager),
            CompressionMethod.KEY_EXTRACTION: KeyExtractionStrategy(model_manager)
        }
    
    async def compress(
        self,
        content: Union[str, List[Message]],
        level: CompressionLevel = CompressionLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """Apply hybrid compression strategy."""
        start_time = time.time()
        
        # Determine best strategy based on content
        best_strategy = await self._select_best_strategy(content, level)
        
        # Apply the selected strategy
        result = await best_strategy.compress(content, level, context)
        
        # Update metadata to indicate hybrid approach
        result.method = CompressionMethod.HYBRID
        result.metadata["selected_strategy"] = best_strategy.method.value
        result.processing_time = time.time() - start_time
        
        self.update_stats(result)
        return result
    
    async def decompress(
        self,
        compressed_data: Union[str, bytes],
        metadata: Dict[str, Any]
    ) -> str:
        """Decompress using the appropriate strategy."""
        strategy_name = metadata.get("selected_strategy")
        if strategy_name and strategy_name in [m.value for m in CompressionMethod]:
            strategy = self.strategies[CompressionMethod(strategy_name)]
            return await strategy.decompress(compressed_data, metadata)
        
        # Fallback
        return str(compressed_data)
    
    async def _select_best_strategy(
        self,
        content: Union[str, List[Message]],
        level: CompressionLevel
    ) -> CompressionStrategy:
        """Select the best compression strategy for the content."""
        if isinstance(content, str):
            text = content
            is_conversation = False
        else:
            text = '\n'.join(msg.get_text_content() for msg in content)
            is_conversation = len(content) > 1
        
        text_length = len(text)
        
        # Strategy selection logic
        if text_length < 500:
            # Short text: use LZ4 for speed
            return self.strategies[CompressionMethod.LZ4]
        elif is_conversation and level >= CompressionLevel.MEDIUM:
            # Conversation with medium+ compression: use summarization
            return self.strategies[CompressionMethod.SUMMARIZATION]
        elif level == CompressionLevel.EXTREME:
            # Extreme compression: use key extraction
            return self.strategies[CompressionMethod.KEY_EXTRACTION]
        else:
            # Default: use summarization
            return self.strategies[CompressionMethod.SUMMARIZATION]


class MessageCompressor:
    """
    Main message compression service.
    
    Provides intelligent compression with multiple strategies,
    automatic strategy selection, and compression history tracking.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.strategies = {
            CompressionMethod.LZ4: LZ4CompressionStrategy(),
            CompressionMethod.SUMMARIZATION: SummarizationStrategy(model_manager),
            CompressionMethod.KEY_EXTRACTION: KeyExtractionStrategy(model_manager),
            CompressionMethod.HYBRID: HybridCompressionStrategy(model_manager)
        }
        
        # Compression history and cache
        self._compression_history: deque = deque(maxlen=1000)
        self._compression_cache: Dict[str, CompressionResult] = {}
        
        # Statistics
        self.stats = {
            "total_compressions": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "average_compression_ratio": 0.0,
            "compression_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("MessageCompressor initialized")
    
    async def compress_messages(
        self,
        messages: List[Message],
        method: CompressionMethod = CompressionMethod.HYBRID,
        level: CompressionLevel = CompressionLevel.MEDIUM,
        session_id: Optional[str] = None
    ) -> MessageHistory:
        """
        Compress a list of messages into a MessageHistory.
        
        Returns:
            MessageHistory with compressed content and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(messages, method, level)
        if cache_key in self._compression_cache:
            self.stats["cache_hits"] += 1
            cached_result = self._compression_cache[cache_key]
            
            return MessageHistory(
                session_id=session_id or "unknown",
                original_messages=messages,
                summary=cached_result.compressed_content,
                compression_ratio=cached_result.compression_ratio,
                original_tokens=sum(msg.estimate_tokens() for msg in messages),
                compressed_tokens=len(cached_result.compressed_content.split()),
                compression_method=cached_result.method.value
            )
        
        self.stats["cache_misses"] += 1
        
        # Perform compression
        strategy = self.strategies[method]
        result = await strategy.compress(messages, level)
        
        # Cache the result
        self._compression_cache[cache_key] = result
        
        # Extract key points and important context
        key_points = await self._extract_key_points(messages, level)
        important_context = await self._extract_important_context(messages)
        
        # Create MessageHistory
        message_history = MessageHistory(
            session_id=session_id or "unknown",
            original_messages=messages,
            summary=result.compressed_content,
            key_points=key_points,
            important_context=important_context,
            compression_ratio=result.compression_ratio,
            original_tokens=sum(msg.estimate_tokens() for msg in messages),
            compressed_tokens=len(str(result.compressed_content).split()),
            compression_method=result.method.value
        )
        
        # Update statistics
        self.stats["total_compressions"] += 1
        self.stats["total_original_bytes"] += result.original_size
        self.stats["total_compressed_bytes"] += result.compressed_size
        self.stats["compression_time"] += result.processing_time
        
        if self.stats["total_original_bytes"] > 0:
            self.stats["average_compression_ratio"] = (
                self.stats["total_compressed_bytes"] / 
                self.stats["total_original_bytes"]
            )
        
        # Add to history
        self._compression_history.append({
            "timestamp": time.time(),
            "session_id": session_id,
            "method": method.value,
            "level": level.value,
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
            "ratio": result.compression_ratio,
            "processing_time": result.processing_time
        })
        
        logger.info(
            f"Compressed {len(messages)} messages",
            method=method.value,
            level=level.value,
            ratio=result.compression_ratio,
            original_size=result.original_size,
            compressed_size=result.compressed_size
        )
        
        return message_history
    
    async def compress_text(
        self,
        text: str,
        method: CompressionMethod = CompressionMethod.HYBRID,
        level: CompressionLevel = CompressionLevel.MEDIUM
    ) -> CompressionResult:
        """Compress raw text."""
        strategy = self.strategies[method]
        return await strategy.compress(text, level)
    
    async def decompress(
        self,
        compressed_data: Union[str, bytes],
        metadata: Dict[str, Any]
    ) -> str:
        """Decompress previously compressed data."""
        method_name = metadata.get("method") or metadata.get("selected_strategy")
        
        if method_name and method_name in [m.value for m in CompressionMethod]:
            method = CompressionMethod(method_name)
            strategy = self.strategies[method]
            return await strategy.decompress(compressed_data, metadata)
        
        # Fallback
        if isinstance(compressed_data, bytes):
            try:
                # Try LZ4 decompression
                return await self.strategies[CompressionMethod.LZ4].decompress(
                    compressed_data, metadata
                )
            except:
                return compressed_data.decode('utf-8', errors='ignore')
        
        return str(compressed_data)
    
    async def _extract_key_points(
        self, 
        messages: List[Message], 
        level: CompressionLevel
    ) -> List[str]:
        """Extract key points from messages."""
        # Simple extraction based on message roles and content
        key_points = []
        
        for msg in messages:
            if msg.role == MessageRole.ASSISTANT:
                # Extract sentences that look like conclusions or important points
                content = msg.get_text_content()
                sentences = content.split('. ')
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Minimum length
                        # Look for important indicators
                        if any(indicator in sentence.lower() for indicator in [
                            'important', 'key', 'main', 'conclude', 'result', 
                            'recommend', 'suggest', 'should', 'must', 'need'
                        ]):
                            key_points.append(sentence)
        
        # Limit based on compression level
        max_points = {
            CompressionLevel.LIGHT: 10,
            CompressionLevel.MEDIUM: 7,
            CompressionLevel.HEAVY: 5,
            CompressionLevel.EXTREME: 3
        }.get(level, 7)
        
        return key_points[:max_points]
    
    async def _extract_important_context(
        self, 
        messages: List[Message]
    ) -> Dict[str, Any]:
        """Extract important context information."""
        context = {
            "participant_count": len(set(msg.role for msg in messages)),
            "message_count": len(messages),
            "time_span": None,
            "topics": [],
            "tools_used": []
        }
        
        if messages:
            start_time = messages[0].timestamp
            end_time = messages[-1].timestamp
            context["time_span"] = (end_time - start_time).total_seconds()
        
        # Extract tools used
        for msg in messages:
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        func_name = tool_call["function"].get("name", "unknown")
                        if func_name not in context["tools_used"]:
                            context["tools_used"].append(func_name)
        
        return context
    
    def _generate_cache_key(
        self,
        messages: List[Message],
        method: CompressionMethod,
        level: CompressionLevel
    ) -> str:
        """Generate cache key for compression results."""
        # Create hash of message contents + method + level
        content_parts = []
        for msg in messages:
            content_parts.append(f"{msg.role}:{msg.get_text_content()}")
        
        content_str = "|".join(content_parts)
        cache_input = f"{content_str}:{method.value}:{level.value}"
        
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self.stats,
            "strategy_stats": {
                method.value: strategy.stats 
                for method, strategy in self.strategies.items()
            },
            "cache_size": len(self._compression_cache),
            "history_size": len(self._compression_history),
            "recent_compressions": list(self._compression_history)[-10:]  # Last 10
        }
    
    def clear_cache(self) -> int:
        """Clear compression cache and return number of cleared items."""
        cache_size = len(self._compression_cache)
        self._compression_cache.clear()
        return cache_size
    
    async def optimize_context(
        self,
        messages: List[Message],
        target_tokens: int,
        session_id: Optional[str] = None
    ) -> List[Message]:
        """
        Optimize context by compressing older messages while preserving recent ones.
        
        Returns:
            Optimized list of messages fitting within target token count
        """
        current_tokens = sum(msg.estimate_tokens() for msg in messages)
        
        if current_tokens <= target_tokens:
            return messages
        
        # Keep recent messages and compress older ones
        optimized_messages = []
        recent_count = max(3, len(messages) // 4)  # Keep at least 3 or 25% recent
        
        recent_messages = messages[-recent_count:]
        older_messages = messages[:-recent_count]
        
        recent_tokens = sum(msg.estimate_tokens() for msg in recent_messages)
        available_tokens = target_tokens - recent_tokens
        
        if older_messages and available_tokens > 100:
            # Compress older messages
            compressed_history = await self.compress_messages(
                older_messages,
                method=CompressionMethod.SUMMARIZATION,
                level=CompressionLevel.MEDIUM,
                session_id=session_id
            )
            
            # Create a compressed message
            compressed_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"[Compressed History]\n{compressed_history.summary}",
                session_id=session_id or "unknown",
                message_type="system_event",
                compressed=True,
                compression_ratio=compressed_history.compression_ratio
            )
            
            optimized_messages.append(compressed_msg)
        
        optimized_messages.extend(recent_messages)
        
        logger.info(
            f"Context optimized",
            original_messages=len(messages),
            original_tokens=current_tokens,
            optimized_messages=len(optimized_messages),
            optimized_tokens=sum(msg.estimate_tokens() for msg in optimized_messages),
            target_tokens=target_tokens
        )
        
        return optimized_messages
