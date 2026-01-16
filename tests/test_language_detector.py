"""Tests for the language detector module."""

import pytest
from unittest.mock import patch, Mock

from app.core.language_detector import LanguageDetector, LanguageResult


class TestLanguageDetector:
    """Test cases for LanguageDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return LanguageDetector(confidence_threshold=0.7)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.confidence_threshold == 0.7
        assert len(detector.supported_languages) > 0
        assert "en" in detector.supported_languages
    
    @patch('app.core.language_detector.detect')
    @patch('app.core.language_detector.detect_langs')
    def test_detect_english(self, mock_detect_langs, mock_detect, detector):
        """Test detecting English text."""
        # Mock langdetect responses
        mock_detect.return_value = "en"
        
        # Mock language probabilities
        mock_lang_prob = Mock()
        mock_lang_prob.lang = "en"
        mock_lang_prob.prob = 0.95
        mock_detect_langs.return_value = [mock_lang_prob]
        
        result = detector.detect_language("This is an English sentence.")
        
        assert isinstance(result, LanguageResult)
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.is_supported is True
    
    @patch('app.core.language_detector.detect')
    @patch('app.core.language_detector.detect_langs')
    def test_detect_spanish(self, mock_detect_langs, mock_detect, detector):
        """Test detecting Spanish text."""
        mock_detect.return_value = "es"
        
        mock_lang_prob = Mock()
        mock_lang_prob.lang = "es"
        mock_lang_prob.prob = 0.88
        mock_detect_langs.return_value = [mock_lang_prob]
        
        result = detector.detect_language("Esta es una oración en español.")
        
        assert result.language == "es"
        assert result.confidence == 0.88
        assert result.is_supported is True
    
    @patch('app.core.language_detector.detect')
    @patch('app.core.language_detector.detect_langs')
    def test_detect_unsupported_language(self, mock_detect_langs, mock_detect, detector):
        """Test detecting unsupported language."""
        mock_detect.return_value = "fi"  # Finnish - not in supported languages
        
        mock_lang_prob = Mock()
        mock_lang_prob.lang = "fi"
        mock_lang_prob.prob = 0.85
        mock_detect_langs.return_value = [mock_lang_prob]
        
        result = detector.detect_language("Tämä on suomenkielinen lause.")
        
        assert result.language == "fi"
        assert result.confidence == 0.85
        assert result.is_supported is False
    
    @patch('app.core.language_detector.detect')
    @patch('app.core.language_detector.detect_langs')
    def test_low_confidence_fallback(self, mock_detect_langs, mock_detect, detector):
        """Test fallback to supported language when confidence is low."""
        mock_detect.return_value = "fi"  # Unsupported
        
        # Mock multiple language probabilities
        mock_lang_prob1 = Mock()
        mock_lang_prob1.lang = "fi"
        mock_lang_prob1.prob = 0.6  # Below threshold
        
        mock_lang_prob2 = Mock()
        mock_lang_prob2.lang = "en"  # Supported
        mock_lang_prob2.prob = 0.75  # Above threshold
        
        mock_detect_langs.return_value = [mock_lang_prob1, mock_lang_prob2]
        
        result = detector.detect_language("Mixed language text.")
        
        # Should fallback to English due to higher confidence and support
        assert result.language == "en"
        assert result.confidence == 0.75
        assert result.is_supported is True
    
    def test_preprocess_text(self, detector):
        """Test text preprocessing."""
        # Test URL removal
        text_with_url = "Check this out https://example.com great stuff!"
        processed = detector._preprocess_text(text_with_url)
        assert "https://example.com" not in processed
        
        # Test email removal
        text_with_email = "Contact me at test@example.com for more info"
        processed = detector._preprocess_text(text_with_email)
        assert "test@example.com" not in processed
        
        # Test whitespace normalization
        text_with_spaces = "Too    many     spaces"
        processed = detector._preprocess_text(text_with_spaces)
        assert "Too many spaces" == processed
        
        # Test short text
        short_text = "Hi"
        processed = detector._preprocess_text(short_text)
        assert processed == ""  # Too short
    
    @patch('app.core.language_detector.detect')
    @patch('app.core.language_detector.detect_langs')
    def test_detect_batch(self, mock_detect_langs, mock_detect, detector):
        """Test batch language detection."""
        texts = [
            "This is English",
            "Esto es español",
            "C'est français"
        ]
        
        # Mock responses for each text
        mock_detect.side_effect = ["en", "es", "fr"]
        
        # Mock language probabilities
        def mock_detect_langs_side_effect(text):
            if "English" in text:
                mock_prob = Mock()
                mock_prob.lang = "en"
                mock_prob.prob = 0.9
                return [mock_prob]
            elif "español" in text:
                mock_prob = Mock()
                mock_prob.lang = "es"
                mock_prob.prob = 0.85
                return [mock_prob]
            else:  # French
                mock_prob = Mock()
                mock_prob.lang = "fr"
                mock_prob.prob = 0.8
                return [mock_prob]
        
        mock_detect_langs.side_effect = mock_detect_langs_side_effect
        
        results = detector.detect_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, LanguageResult) for r in results)
        assert results[0].language == "en"
        assert results[1].language == "es"
        assert results[2].language == "fr"
    
    def test_get_language_name(self, detector):
        """Test getting language names."""
        assert detector.get_language_name("en") == "English"
        assert detector.get_language_name("es") == "Spanish"
        assert detector.get_language_name("fr") == "French"
        assert detector.get_language_name("unknown") == "UNKNOWN"
    
    def test_is_supported(self, detector):
        """Test checking if language is supported."""
        assert detector.is_supported("en") is True
        assert detector.is_supported("es") is True
        assert detector.is_supported("fi") is False  # Not in supported list
    
    def test_get_supported_languages(self, detector):
        """Test getting supported languages."""
        supported = detector.get_supported_languages()
        
        assert isinstance(supported, dict)
        assert "en" in supported
        assert supported["en"] == "English"
        assert len(supported) > 0
    
    def test_get_all_detectable_languages(self, detector):
        """Test getting all detectable languages."""
        all_langs = detector.get_all_detectable_languages()
        
        assert isinstance(all_langs, dict)
        assert len(all_langs) > len(detector.supported_languages)
        assert "en" in all_langs
        assert "fi" in all_langs  # Should include non-supported languages
    
    @patch('app.core.language_detector.detect')
    def test_error_handling(self, mock_detect, detector):
        """Test error handling during detection."""
        from langdetect.lang_detect_exception import LangDetectException
        
        # Mock detection failure
        mock_detect.side_effect = LangDetectException("Detection failed", "test")
        
        result = detector.detect_language("Some text")
        
        # Should return default English result
        assert result.language == "en"
        assert result.confidence == 0.0
        assert result.is_supported is True
    
    @pytest.mark.parametrize("text,expected_lang", [
        ("Hello world", "en"),
        ("Hola mundo", "es"),
        ("Bonjour le monde", "fr"),
        ("Hallo Welt", "de"),
        ("你好世界", "zh")
    ])
    def test_language_examples(self, text, expected_lang, detector):
        """Test language detection with example texts."""
        with patch('app.core.language_detector.detect') as mock_detect, \
             patch('app.core.language_detector.detect_langs') as mock_detect_langs:
            
            mock_detect.return_value = expected_lang
            
            mock_lang_prob = Mock()
            mock_lang_prob.lang = expected_lang
            mock_lang_prob.prob = 0.9
            mock_detect_langs.return_value = [mock_lang_prob]
            
            result = detector.detect_language(text)
            assert result.language == expected_lang
