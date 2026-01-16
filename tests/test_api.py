"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.api.main import app
from app.core.multilingual_analyzer import AnalysisResult


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_analyzer():
    """Create mock analyzer for testing."""
    analyzer = Mock()
    
    # Mock analysis result
    mock_result = AnalysisResult(
        original_text="Test text",
        detected_language="en",
        language_confidence=0.95,
        language_alternatives=[("es", 0.03), ("fr", 0.02)],
        translation_needed=False,
        translated_text=None,
        translation_confidence=None,
        preprocessed_text="Test text",
        preprocessing_stats=None,
        sentiment="positive",
        sentiment_confidence=0.85,
        sentiment_scores={"positive": 0.85, "negative": 0.1, "neutral": 0.05},
        processing_time=0.123,
        model_used="test-model",
        cached_translation=False
    )
    
    analyzer.analyze_text.return_value = mock_result
    analyzer.analyze_batch.return_value = [mock_result]
    analyzer.language_detector.get_language_name.return_value = "English"
    analyzer.language_detector.is_supported.return_value = True
    analyzer.enable_translation = True
    analyzer.enable_preprocessing = True
    
    return analyzer


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["service"] == "Multilingual Sentiment Analysis Tool"
    
    @patch('app.api.routes.health.analyzer', None)
    def test_health_check_no_analyzer(self, client):
        """Test health check when analyzer is not loaded."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
    
    @patch('app.api.routes.health.analyzer')
    def test_health_check_with_analyzer(self, mock_analyzer_patch, client):
        """Test health check when analyzer is loaded."""
        # Mock analyzer
        mock_analyzer_patch.language_detector.get_supported_languages.return_value = {"en": "English"}
        mock_analyzer_patch.get_system_info.return_value = {"model": "test"}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "supported_languages" in data


class TestSentimentEndpoints:
    """Test sentiment analysis endpoints."""
    
    @patch('app.api.routes.sentiment.analyzer')
    def test_analyze_text_success(self, mock_analyzer_patch, client, mock_analyzer):
        """Test successful text analysis."""
        mock_analyzer_patch.return_value = mock_analyzer
        mock_analyzer_patch.analyze_text = mock_analyzer.analyze_text
        mock_analyzer_patch.language_detector = mock_analyzer.language_detector
        mock_analyzer_patch.enable_translation = True
        mock_analyzer_patch.enable_preprocessing = True
        
        response = client.post(
            "/analyze/",
            json={
                "text": "I love this product!",
                "enable_translation": True,
                "enable_preprocessing": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.85
        assert data["original_text"] == "Test text"
        assert "language" in data
        assert "translation" in data
        assert "preprocessing" in data
        assert "scores" in data
    
    def test_analyze_text_validation_error(self, client):
        """Test text analysis with validation error."""
        response = client.post(
            "/analyze/",
            json={
                "text": "",  # Empty text should fail validation
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "validation_error"
    
    def test_analyze_text_missing_field(self, client):
        """Test text analysis with missing required field."""
        response = client.post(
            "/analyze/",
            json={}  # Missing text field
        )
        
        assert response.status_code == 422
    
    @patch('app.api.routes.sentiment.analyzer')
    def test_analyze_batch_success(self, mock_analyzer_patch, client, mock_analyzer):
        """Test successful batch analysis."""
        mock_analyzer_patch.return_value = mock_analyzer
        mock_analyzer_patch.analyze_batch = mock_analyzer.analyze_batch
        mock_analyzer_patch.language_detector = mock_analyzer.language_detector
        mock_analyzer_patch.enable_translation = True
        mock_analyzer_patch.enable_preprocessing = True
        
        response = client.post(
            "/analyze/batch",
            json={
                "texts": ["Great product!", "Terrible service"],
                "enable_translation": True,
                "enable_preprocessing": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "summary" in data
        assert "total_processing_time" in data
        assert len(data["results"]) == 1  # Mock returns single result
        assert "sentiment_distribution" in data["summary"]
    
    def test_analyze_batch_validation_error(self, client):
        """Test batch analysis with validation error."""
        response = client.post(
            "/analyze/batch",
            json={
                "texts": [],  # Empty list should fail validation
            }
        )
        
        assert response.status_code == 422
    
    def test_analyze_batch_too_many_texts(self, client):
        """Test batch analysis with too many texts."""
        response = client.post(
            "/analyze/batch",
            json={
                "texts": ["text"] * 101,  # Exceeds limit of 100
            }
        )
        
        assert response.status_code == 422


class TestLanguageEndpoints:
    """Test language detection endpoints."""
    
    @patch('app.api.routes.language.language_detector')
    def test_detect_language_success(self, mock_detector, client):
        """Test successful language detection."""
        from app.core.language_detector import LanguageResult
        
        mock_result = LanguageResult(
            language="en",
            confidence=0.95,
            is_supported=True,
            alternatives=[("es", 0.03)]
        )
        
        mock_detector.detect_language.return_value = mock_result
        mock_detector.get_language_name.return_value = "English"
        
        response = client.post(
            "/language/detect",
            json={"text": "This is English text"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["language"]["code"] == "en"
        assert data["language"]["confidence"] == 0.95
        assert data["language"]["is_supported"] is True
    
    def test_detect_language_validation_error(self, client):
        """Test language detection with validation error."""
        response = client.post(
            "/language/detect",
            json={"text": ""}  # Empty text
        )
        
        assert response.status_code == 422
    
    @patch('app.api.routes.language.language_detector')
    def test_detect_languages_batch_success(self, mock_detector, client):
        """Test successful batch language detection."""
        from app.core.language_detector import LanguageResult
        
        mock_results = [
            LanguageResult(language="en", confidence=0.95, is_supported=True, alternatives=[]),
            LanguageResult(language="es", confidence=0.88, is_supported=True, alternatives=[])
        ]
        
        mock_detector.detect_batch.return_value = mock_results
        mock_detector.get_language_name.side_effect = lambda x: {"en": "English", "es": "Spanish"}[x]
        
        response = client.post(
            "/language/detect/batch",
            json={"texts": ["English text", "Texto español"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 2
        assert "summary" in data
        assert data["summary"]["total_texts"] == 2
    
    @patch('app.api.routes.language.language_detector')
    def test_get_supported_languages(self, mock_detector, client):
        """Test getting supported languages."""
        mock_detector.get_supported_languages.return_value = {
            "en": "English",
            "es": "Spanish",
            "fr": "French"
        }
        
        response = client.get("/language/supported")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "languages" in data
        assert "count" in data
        assert data["count"] == 3
        assert "en" in data["languages"]


class TestErrorHandling:
    """Test error handling."""
    
    @patch('app.api.routes.sentiment.analyzer')
    def test_internal_server_error(self, mock_analyzer_patch, client):
        """Test internal server error handling."""
        # Mock analyzer to raise exception
        mock_analyzer_patch.analyze_text.side_effect = Exception("Internal error")
        
        response = client.post(
            "/analyze/",
            json={"text": "Test text"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "internal_error"
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling."""
        response = client.post(
            "/analyze/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API."""
    
    @pytest.mark.slow
    def test_full_analysis_pipeline(self, client):
        """Test full analysis pipeline (requires real models)."""
        # This would test with real models - skip in unit tests
        pytest.skip("Integration test requires real models")
    
    def test_api_documentation(self, client):
        """Test API documentation endpoints."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
