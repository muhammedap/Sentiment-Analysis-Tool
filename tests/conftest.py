"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_texts():
    """Sample texts for testing in multiple languages."""
    return {
        "english": [
            "I love this product! It's amazing.",
            "This is terrible. I hate it.",
            "The weather is okay today.",
            "Absolutely fantastic experience!",
            "Worst service I've ever received."
        ],
        "spanish": [
            "Me encanta este producto! Es increíble.",
            "Esto es terrible. Lo odio.",
            "El clima está bien hoy.",
            "¡Experiencia absolutamente fantástica!",
            "El peor servicio que he recibido."
        ],
        "french": [
            "J'adore ce produit! C'est incroyable.",
            "C'est terrible. Je le déteste.",
            "Le temps est correct aujourd'hui.",
            "Expérience absolument fantastique!",
            "Le pire service que j'aie jamais reçu."
        ],
        "german": [
            "Ich liebe dieses Produkt! Es ist erstaunlich.",
            "Das ist schrecklich. Ich hasse es.",
            "Das Wetter ist heute okay.",
            "Absolut fantastische Erfahrung!",
            "Der schlechteste Service, den ich je erhalten habe."
        ],
        "chinese": [
            "我喜欢这个产品！太棒了。",
            "这太糟糕了。我讨厌它。",
            "今天天气还可以。",
            "绝对棒极了的体验！",
            "我收到过的最糟糕的服务。"
        ]
    }


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV file for testing."""
    csv_file = temp_dir / "test_data.csv"
    
    import pandas as pd
    
    data = {
        "text": [
            "I love this product!",
            "This is terrible.",
            "The weather is okay.",
            "Amazing experience!",
            "Worst service ever."
        ],
        "expected_sentiment": [
            "positive",
            "negative", 
            "neutral",
            "positive",
            "negative"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    return csv_file


@pytest.fixture
def mock_model_responses():
    """Mock responses for different sentiment analysis scenarios."""
    return {
        "positive": [
            {"label": "LABEL_2", "score": 0.85},
            {"label": "LABEL_1", "score": 0.10},
            {"label": "LABEL_0", "score": 0.05}
        ],
        "negative": [
            {"label": "LABEL_0", "score": 0.80},
            {"label": "LABEL_1", "score": 0.15},
            {"label": "LABEL_2", "score": 0.05}
        ],
        "neutral": [
            {"label": "LABEL_1", "score": 0.70},
            {"label": "LABEL_2", "score": 0.20},
            {"label": "LABEL_0", "score": 0.10}
        ]
    }


@pytest.fixture
def mock_language_responses():
    """Mock responses for language detection scenarios."""
    return {
        "english": {
            "language": "en",
            "confidence": 0.95,
            "alternatives": [("es", 0.03), ("fr", 0.02)]
        },
        "spanish": {
            "language": "es", 
            "confidence": 0.88,
            "alternatives": [("en", 0.08), ("pt", 0.04)]
        },
        "french": {
            "language": "fr",
            "confidence": 0.92,
            "alternatives": [("en", 0.05), ("es", 0.03)]
        },
        "german": {
            "language": "de",
            "confidence": 0.89,
            "alternatives": [("en", 0.07), ("nl", 0.04)]
        },
        "chinese": {
            "language": "zh",
            "confidence": 0.96,
            "alternatives": [("ja", 0.03), ("ko", 0.01)]
        }
    }


@pytest.fixture
def mock_translation_responses():
    """Mock responses for translation scenarios."""
    return {
        "spanish_to_english": {
            "original": "Me encanta este producto",
            "translated": "I love this product",
            "confidence": 0.92
        },
        "french_to_english": {
            "original": "J'adore ce produit",
            "translated": "I love this product", 
            "confidence": 0.89
        },
        "german_to_english": {
            "original": "Ich liebe dieses Produkt",
            "translated": "I love this product",
            "confidence": 0.94
        }
    }


@pytest.fixture(autouse=True)
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "DEBUG": "True",
        "LOG_LEVEL": "DEBUG",
        "DEFAULT_MODEL": "test-model",
        "CACHE_DIR": "/tmp/test_cache",
        "MAX_LENGTH": "512",
        "BATCH_SIZE": "16",
        "API_HOST": "127.0.0.1",
        "API_PORT": "8000"
    }):
        yield


@pytest.fixture
def mock_torch():
    """Mock torch for testing without GPU requirements."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.cuda.device_count', return_value=0):
        yield


# Performance testing fixtures
@pytest.fixture
def performance_texts():
    """Generate texts for performance testing."""
    import random
    import string
    
    def generate_text(length):
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))
    
    return {
        "short": [generate_text(50) for _ in range(100)],
        "medium": [generate_text(200) for _ in range(50)], 
        "long": [generate_text(1000) for _ in range(10)]
    }


@pytest.fixture
def benchmark_data():
    """Benchmark data for accuracy testing."""
    return {
        "positive_samples": [
            "I absolutely love this product!",
            "This is the best thing ever!",
            "Amazing quality and great service!",
            "Fantastic experience, highly recommend!",
            "Perfect! Exactly what I needed."
        ],
        "negative_samples": [
            "This is absolutely terrible!",
            "Worst product I've ever bought!",
            "Completely disappointed and frustrated!",
            "Horrible quality and terrible service!",
            "I hate this so much, total waste of money!"
        ],
        "neutral_samples": [
            "The product arrived on time.",
            "It's an okay product, nothing special.",
            "The weather is cloudy today.",
            "I received the package yesterday.",
            "The meeting is scheduled for 3 PM."
        ]
    }


# Skip markers for different test types
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers."""
    for item in items:
        # Skip slow tests by default
        if "slow" in item.keywords and not config.getoption("--runslow", default=False):
            item.add_marker(pytest.mark.skip(reason="need --runslow option to run"))
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(pytest.mark.skip(reason="GPU not available"))
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--rungpu",
        action="store_true",
        default=False,
        help="run GPU tests"
    )
