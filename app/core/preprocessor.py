"""Text preprocessing pipeline for multilingual sentiment analysis."""

import re
import logging
import unicodedata
from typing import List, Optional, Dict, Any
import html

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Comprehensive text preprocessing for multilingual sentiment analysis."""
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_mentions: bool = False,
                 remove_hashtags: bool = False,
                 normalize_unicode: bool = True,
                 remove_extra_whitespace: bool = True,
                 min_length: int = 3,
                 max_length: Optional[int] = None):
        """Initialize preprocessor.
        
        Args:
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            normalize_unicode: Normalize unicode characters
            remove_extra_whitespace: Remove extra whitespace
            min_length: Minimum text length after preprocessing
            max_length: Maximum text length (truncate if longer)
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for text cleaning."""
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        
        # Mention pattern (@username)
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern (#hashtag)
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # HTML entities pattern
        self.html_pattern = re.compile(r'&[a-zA-Z0-9#]+;')
        
        # Repeated punctuation pattern
        self.repeated_punct_pattern = re.compile(r'([.!?]){2,}')
        
        # Non-printable characters pattern
        self.non_printable_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]')
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Remove non-printable characters
        text = self.non_printable_pattern.sub('', text)
        
        return text
    
    def _clean_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)
    
    def _clean_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub('', text)
    
    def _clean_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return self.mention_pattern.sub('', text)
    
    def _clean_hashtags(self, text: str) -> str:
        """Remove #hashtags from text."""
        return self.hashtag_pattern.sub('', text)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize repeated punctuation."""
        # Replace repeated punctuation with single occurrence
        text = self.repeated_punct_pattern.sub(r'\1', text)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean extra whitespace."""
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Start with original text
        processed_text = text
        
        # Clean HTML
        processed_text = self._clean_html(processed_text)
        
        # Normalize unicode if enabled
        if self.normalize_unicode:
            processed_text = self._normalize_unicode(processed_text)
        
        # Remove URLs if enabled
        if self.remove_urls:
            processed_text = self._clean_urls(processed_text)
        
        # Remove emails if enabled
        if self.remove_emails:
            processed_text = self._clean_emails(processed_text)
        
        # Remove mentions if enabled
        if self.remove_mentions:
            processed_text = self._clean_mentions(processed_text)
        
        # Remove hashtags if enabled
        if self.remove_hashtags:
            processed_text = self._clean_hashtags(processed_text)
        
        # Normalize punctuation
        processed_text = self._normalize_punctuation(processed_text)
        
        # Clean whitespace if enabled
        if self.remove_extra_whitespace:
            processed_text = self._clean_whitespace(processed_text)
        
        # Check minimum length
        if len(processed_text) < self.min_length:
            logger.warning(f"Text too short after preprocessing: '{processed_text}'")
            return ""
        
        # Truncate if maximum length is set
        if self.max_length and len(processed_text) > self.max_length:
            processed_text = processed_text[:self.max_length].strip()
            logger.info(f"Text truncated to {self.max_length} characters")
        
        return processed_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def get_preprocessing_stats(self, original_text: str, processed_text: str) -> Dict[str, Any]:
        """Get preprocessing statistics.
        
        Args:
            original_text: Original text
            processed_text: Processed text
            
        Returns:
            Dictionary with preprocessing statistics
        """
        return {
            "original_length": len(original_text),
            "processed_length": len(processed_text),
            "length_reduction": len(original_text) - len(processed_text),
            "reduction_percentage": (
                (len(original_text) - len(processed_text)) / len(original_text) * 100
                if len(original_text) > 0 else 0
            ),
            "is_empty_after_processing": len(processed_text) == 0,
            "meets_min_length": len(processed_text) >= self.min_length
        }
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """Validate text for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            Validation results
        """
        if not text:
            return {
                "is_valid": False,
                "issues": ["Text is empty or None"],
                "recommendations": ["Provide non-empty text"]
            }
        
        if not isinstance(text, str):
            return {
                "is_valid": False,
                "issues": ["Text is not a string"],
                "recommendations": ["Convert to string before processing"]
            }
        
        issues = []
        recommendations = []
        
        # Check length
        if len(text) < self.min_length:
            issues.append(f"Text too short (minimum {self.min_length} characters)")
            recommendations.append("Provide longer text for better analysis")
        
        if self.max_length and len(text) > self.max_length:
            issues.append(f"Text too long (maximum {self.max_length} characters)")
            recommendations.append("Text will be truncated")
        
        # Check for non-printable characters
        non_printable_count = len(self.non_printable_pattern.findall(text))
        if non_printable_count > 0:
            issues.append(f"Contains {non_printable_count} non-printable characters")
            recommendations.append("Non-printable characters will be removed")
        
        # Check encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("Text contains invalid unicode characters")
            recommendations.append("Fix encoding issues")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "character_count": len(text),
            "word_count": len(text.split()),
            "non_printable_count": non_printable_count
        }
