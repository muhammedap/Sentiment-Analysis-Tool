"""Translation service for unsupported languages."""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None

from ..utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Translation result."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    processing_time: float
    cached: bool = False


class TranslationCache:
    """Simple in-memory translation cache."""
    
    def __init__(self, ttl: int = 3600):
        """Initialize cache.
        
        Args:
            ttl: Time to live in seconds
        """
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key."""
        content = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation."""
        key = self._get_cache_key(text, source_lang, target_lang)
        if key in self.cache:
            translation, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return translation
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, text: str, source_lang: str, target_lang: str, translation: str) -> None:
        """Cache translation."""
        key = self._get_cache_key(text, source_lang, target_lang)
        self.cache[key] = (translation, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class TranslationService:
    """Translation service with caching and error handling."""
    
    def __init__(self, cache_ttl: int = None):
        """Initialize translation service.

        Args:
            cache_ttl: Cache time to live in seconds
        """
        self.cache_ttl = cache_ttl or settings.translation_cache_ttl
        self.cache = TranslationCache(self.cache_ttl)

        if TRANSLATOR_AVAILABLE:
            self.translator = GoogleTranslator(source='auto', target='en')
        else:
            self.translator = None
            logger.warning("Translation service not available - deep-translator not installed")

        self.supported_languages = set(settings.supported_languages)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _choose_target_language(self, source_language: str) -> str:
        """Choose best target language for translation.
        
        Args:
            source_language: Source language code
            
        Returns:
            Target language code
        """
        # If source is already supported, no translation needed
        if source_language in self.supported_languages:
            return source_language
        
        # Default to English as it's most widely supported
        return "en"
    
    def translate_text(
        self, 
        text: str, 
        source_language: str,
        target_language: Optional[str] = None
    ) -> TranslationResult:
        """Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code (auto-selected if None)
            
        Returns:
            TranslationResult object
        """
        start_time = time.time()
        
        # Choose target language if not specified
        if target_language is None:
            target_language = self._choose_target_language(source_language)
        
        # If source and target are the same, no translation needed
        if source_language == target_language:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=1.0,
                processing_time=time.time() - start_time,
                cached=False
            )
        
        # Check cache first
        cached_translation = self.cache.get(text, source_language, target_language)
        if cached_translation:
            return TranslationResult(
                original_text=text,
                translated_text=cached_translation,
                source_language=source_language,
                target_language=target_language,
                confidence=0.9,  # Assume high confidence for cached results
                processing_time=time.time() - start_time,
                cached=True
            )
        
        try:
            if not self.translator:
                # Return original text if translator not available
                return TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    cached=False
                )

            # Apply rate limiting
            self._rate_limit()

            # Perform translation using deep-translator
            self.translator.source = source_language
            self.translator.target = target_language
            translated_text = self.translator.translate(text)

            confidence = 0.8  # Default confidence for deep-translator
            
            # Cache the result
            self.cache.set(text, source_language, target_language, translated_text)
            
            processing_time = time.time() - start_time
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence=confidence,
                processing_time=processing_time,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            processing_time = time.time() - start_time
            
            # Return original text on failure
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.0,
                processing_time=processing_time,
                cached=False
            )
    
    def translate_batch(
        self, 
        texts: List[str], 
        source_languages: List[str],
        target_language: Optional[str] = None
    ) -> List[TranslationResult]:
        """Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source_languages: List of source language codes
            target_language: Target language code (auto-selected if None)
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        
        for text, source_lang in zip(texts, source_languages):
            result = self.translate_text(text, source_lang, target_language)
            results.append(result)
            
        return results
    
    def needs_translation(self, language_code: str) -> bool:
        """Check if language needs translation.
        
        Args:
            language_code: Language code to check
            
        Returns:
            True if translation is needed
        """
        return language_code not in self.supported_languages
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": self.cache.size(),
            "cache_ttl": self.cache_ttl
        }
    
    def clear_cache(self) -> None:
        """Clear translation cache."""
        self.cache.clear()
        logger.info("Translation cache cleared")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(self.supported_languages)
