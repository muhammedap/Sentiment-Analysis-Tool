"""Tests for the sentiment analyzer module."""

import pytest
import torch
from unittest.mock import Mock, patch

from app.core.sentiment_analyzer import MultilingualSentimentAnalyzer, SentimentResult


class TestMultilingualSentimentAnalyzer:
    """Test cases for MultilingualSentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        # Mock the model loading to avoid downloading during tests
        with patch('app.core.sentiment_analyzer.AutoTokenizer'), \
             patch('app.core.sentiment_analyzer.AutoModelForSequenceClassification'), \
             patch('app.core.sentiment_analyzer.pipeline') as mock_pipeline:
            
            # Mock pipeline return value
            mock_pipeline.return_value = Mock()
            
            analyzer = MultilingualSentimentAnalyzer(
                model_name="test-model",
                device="cpu"
            )
            
            # Mock the pipeline attribute
            analyzer.pipeline = Mock()
            
            return analyzer
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.model_name == "test-model"
        assert analyzer.device == "cpu"
        assert analyzer.pipeline is not None
    
    def test_analyze_text_positive(self, analyzer):
        """Test analyzing positive text."""
        # Mock pipeline response
        analyzer.pipeline.return_value = [[
            {"label": "LABEL_2", "score": 0.8},
            {"label": "LABEL_1", "score": 0.15},
            {"label": "LABEL_0", "score": 0.05}
        ]]
        
        result = analyzer.analyze_text("I love this product!")
        
        assert isinstance(result, SentimentResult)
        assert result.sentiment == "positive"
        assert result.confidence == 0.8
        assert result.text == "I love this product!"
        assert "positive" in result.scores
        assert "negative" in result.scores
        assert "neutral" in result.scores
    
    def test_analyze_text_negative(self, analyzer):
        """Test analyzing negative text."""
        analyzer.pipeline.return_value = [[
            {"label": "LABEL_0", "score": 0.9},
            {"label": "LABEL_1", "score": 0.08},
            {"label": "LABEL_2", "score": 0.02}
        ]]
        
        result = analyzer.analyze_text("This is terrible!")
        
        assert result.sentiment == "negative"
        assert result.confidence == 0.9
    
    def test_analyze_text_neutral(self, analyzer):
        """Test analyzing neutral text."""
        analyzer.pipeline.return_value = [[
            {"label": "LABEL_1", "score": 0.7},
            {"label": "LABEL_2", "score": 0.2},
            {"label": "LABEL_0", "score": 0.1}
        ]]
        
        result = analyzer.analyze_text("This is a statement.")
        
        assert result.sentiment == "neutral"
        assert result.confidence == 0.7
    
    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        texts = ["Great product!", "Terrible service", "It's okay"]
        
        analyzer.pipeline.return_value = [
            [{"label": "LABEL_2", "score": 0.9}, {"label": "LABEL_1", "score": 0.08}, {"label": "LABEL_0", "score": 0.02}],
            [{"label": "LABEL_0", "score": 0.85}, {"label": "LABEL_1", "score": 0.1}, {"label": "LABEL_2", "score": 0.05}],
            [{"label": "LABEL_1", "score": 0.6}, {"label": "LABEL_2", "score": 0.25}, {"label": "LABEL_0", "score": 0.15}]
        ]
        
        results = analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"
        assert results[2].sentiment == "neutral"
    
    def test_analyze_empty_text(self, analyzer):
        """Test analyzing empty text."""
        result = analyzer.analyze_text("")
        
        # Should handle gracefully and return neutral
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
    
    def test_analyze_long_text(self, analyzer):
        """Test analyzing very long text."""
        long_text = "This is a test. " * 1000  # Very long text
        
        analyzer.pipeline.return_value = [[
            {"label": "LABEL_1", "score": 0.6},
            {"label": "LABEL_2", "score": 0.3},
            {"label": "LABEL_0", "score": 0.1}
        ]]
        
        result = analyzer.analyze_text(long_text)
        
        assert isinstance(result, SentimentResult)
        assert result.sentiment == "neutral"
    
    def test_get_model_info(self, analyzer):
        """Test getting model information."""
        info = analyzer.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert info["model_name"] == "test-model"
        assert info["device"] == "cpu"
    
    def test_normalize_sentiment_labels(self, analyzer):
        """Test sentiment label normalization."""
        # Test standard labels
        results = [
            {"label": "LABEL_0", "score": 0.3},
            {"label": "LABEL_1", "score": 0.4},
            {"label": "LABEL_2", "score": 0.3}
        ]
        
        normalized = analyzer._normalize_sentiment_labels(results)
        
        assert "negative" in normalized
        assert "neutral" in normalized
        assert "positive" in normalized
        assert normalized["negative"] == 0.3
        assert normalized["neutral"] == 0.4
        assert normalized["positive"] == 0.3
    
    def test_error_handling(self, analyzer):
        """Test error handling during analysis."""
        # Mock pipeline to raise exception
        analyzer.pipeline.side_effect = Exception("Model error")
        
        result = analyzer.analyze_text("Test text")
        
        # Should return neutral result on error
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
        assert result.text == "Test text"


@pytest.mark.integration
class TestSentimentAnalyzerIntegration:
    """Integration tests for sentiment analyzer."""
    
    @pytest.mark.slow
    def test_real_model_loading(self):
        """Test loading a real model (slow test)."""
        # Skip if no GPU available and model is large
        if not torch.cuda.is_available():
            pytest.skip("Skipping real model test without GPU")
        
        analyzer = MultilingualSentimentAnalyzer(
            model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        
        # Test with simple text
        result = analyzer.analyze_text("I love this!")
        
        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0 <= result.confidence <= 1
    
    @pytest.mark.parametrize("text,expected_sentiment", [
        ("I love this product!", "positive"),
        ("This is absolutely terrible!", "negative"),
        ("The weather is okay today.", "neutral"),
        ("Amazing experience!", "positive"),
        ("Worst service ever!", "negative")
    ])
    def test_sentiment_examples(self, text, expected_sentiment):
        """Test sentiment analysis with example texts."""
        # This would require a real model, so we'll mock it
        with patch('app.core.sentiment_analyzer.AutoTokenizer'), \
             patch('app.core.sentiment_analyzer.AutoModelForSequenceClassification'), \
             patch('app.core.sentiment_analyzer.pipeline') as mock_pipeline:
            
            # Mock appropriate response based on expected sentiment
            if expected_sentiment == "positive":
                mock_response = [[
                    {"label": "LABEL_2", "score": 0.8},
                    {"label": "LABEL_1", "score": 0.15},
                    {"label": "LABEL_0", "score": 0.05}
                ]]
            elif expected_sentiment == "negative":
                mock_response = [[
                    {"label": "LABEL_0", "score": 0.8},
                    {"label": "LABEL_1", "score": 0.15},
                    {"label": "LABEL_2", "score": 0.05}
                ]]
            else:  # neutral
                mock_response = [[
                    {"label": "LABEL_1", "score": 0.8},
                    {"label": "LABEL_2", "score": 0.15},
                    {"label": "LABEL_0", "score": 0.05}
                ]]
            
            mock_pipeline.return_value = Mock()
            analyzer = MultilingualSentimentAnalyzer()
            analyzer.pipeline.return_value = mock_response
            
            result = analyzer.analyze_text(text)
            assert result.sentiment == expected_sentiment
