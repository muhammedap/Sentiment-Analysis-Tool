"""Language detection module for multilingual text analysis."""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from ..utils.config import settings

logger = logging.getLogger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0


@dataclass
class LanguageResult:
    """Language detection result."""
    language: str
    confidence: float
    is_supported: bool
    alternatives: List[Tuple[str, float]]


class LanguageDetector:
    """Language detection service."""
    
    # Language code mappings
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "pt": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ja": "Japanese",
        "ar": "Arabic",
        "nl": "Dutch",
        "ko": "Korean",
        "hi": "Hindi",
        "tr": "Turkish",
        "pl": "Polish",
        "sv": "Swedish",
        "da": "Danish",
        "no": "Norwegian",
        "fi": "Finnish",
        "cs": "Czech",
        "hu": "Hungarian",
        "ro": "Romanian",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "et": "Estonian",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "mt": "Maltese",
        "cy": "Welsh",
        "ga": "Irish",
        "eu": "Basque",
        "ca": "Catalan",
        "gl": "Galician",
        "th": "Thai",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "ms": "Malay",
        "tl": "Filipino",
        "sw": "Swahili",
        "af": "Afrikaans",
        "sq": "Albanian",
        "az": "Azerbaijani",
        "be": "Belarusian",
        "bn": "Bengali",
        "bs": "Bosnian",
        "eo": "Esperanto",
        "fa": "Persian",
        "gu": "Gujarati",
        "he": "Hebrew",
        "is": "Icelandic",
        "ka": "Georgian",
        "kn": "Kannada",
        "kk": "Kazakh",
        "ky": "Kyrgyz",
        "la": "Latin",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mr": "Marathi",
        "ne": "Nepali",
        "pa": "Punjabi",
        "si": "Sinhala",
        "ta": "Tamil",
        "te": "Telugu",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek"
    }
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize language detector.
        
        Args:
            confidence_threshold: Minimum confidence for language detection
        """
        self.confidence_threshold = confidence_threshold
        self.supported_languages = set(settings.supported_languages)
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better language detection.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts (less than 3 characters)
        if len(text.strip()) < 3:
            return ""
            
        return text
    
    def detect_language(self, text: str) -> LanguageResult:
        """Detect language of input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageResult with detection results
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        if not processed_text:
            logger.warning("Text too short or empty after preprocessing")
            return LanguageResult(
                language="en",  # Default to English
                confidence=0.0,
                is_supported=True,
                alternatives=[]
            )
        
        try:
            # Get primary language detection
            primary_lang = detect(processed_text)
            
            # Get all possible languages with probabilities
            lang_probs = detect_langs(processed_text)
            
            # Find confidence for primary language
            primary_confidence = 0.0
            alternatives = []
            
            for lang_prob in lang_probs:
                if lang_prob.lang == primary_lang:
                    primary_confidence = lang_prob.prob
                else:
                    alternatives.append((lang_prob.lang, lang_prob.prob))
            
            # Sort alternatives by confidence
            alternatives.sort(key=lambda x: x[1], reverse=True)
            
            # Check if language is supported
            is_supported = primary_lang in self.supported_languages
            
            # If confidence is too low, try to find a supported alternative
            if primary_confidence < self.confidence_threshold:
                for alt_lang, alt_conf in alternatives:
                    if alt_lang in self.supported_languages and alt_conf > self.confidence_threshold:
                        logger.info(f"Switching from {primary_lang} to {alt_lang} (higher confidence)")
                        primary_lang = alt_lang
                        primary_confidence = alt_conf
                        is_supported = True
                        break
            
            return LanguageResult(
                language=primary_lang,
                confidence=primary_confidence,
                is_supported=is_supported,
                alternatives=alternatives[:5]  # Top 5 alternatives
            )
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            # Return default result
            return LanguageResult(
                language="en",  # Default to English
                confidence=0.0,
                is_supported=True,
                alternatives=[]
            )
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return LanguageResult(
                language="en",
                confidence=0.0,
                is_supported=True,
                alternatives=[]
            )
    
    def detect_batch(self, texts: List[str]) -> List[LanguageResult]:
        """Detect languages for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of LanguageResult objects
        """
        results = []
        for text in texts:
            result = self.detect_language(text)
            results.append(result)
        return results
    
    def get_language_name(self, language_code: str) -> str:
        """Get human-readable language name.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Human-readable language name
        """
        return self.LANGUAGE_NAMES.get(language_code, language_code.upper())
    
    def is_supported(self, language_code: str) -> bool:
        """Check if language is supported.
        
        Args:
            language_code: ISO language code
            
        Returns:
            True if language is supported
        """
        return language_code in self.supported_languages
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages with names.
        
        Returns:
            Dictionary mapping language codes to names
        """
        return {
            code: self.get_language_name(code) 
            for code in self.supported_languages
        }
    
    def get_all_detectable_languages(self) -> Dict[str, str]:
        """Get all detectable languages with names.
        
        Returns:
            Dictionary mapping language codes to names
        """
        return self.LANGUAGE_NAMES.copy()
