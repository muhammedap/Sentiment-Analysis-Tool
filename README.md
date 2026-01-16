# 🌍 Multilingual Sentiment Analysis Tool

A comprehensive sentiment analysis tool that can analyze sentiment in user reviews across multiple languages using state-of-the-art transformer models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B.svg)](https://multilingual-sentiment-analysis.streamlit.app/)

## 🚀 **[Try the Live Demo](https://multilingual-sentiment-analysis.streamlit.app/)**

## ✨ Features

- **🌐 Multilingual Support**: Supports 10+ major languages with automatic detection
- **🤖 Advanced Models**: Uses XLM-RoBERTa/mBERT for accurate sentiment classification
- **🔍 Language Detection**: Automatic language identification with confidence scoring
- **🔄 Translation Fallback**: Automatic translation for unsupported languages
- **📊 Batch Processing**: Analyze multiple reviews via CSV upload or bulk text input
- **🖥️ Web Interface**: User-friendly Streamlit interface with analytics dashboard
- **🚀 REST API**: FastAPI backend for programmatic access
- **📈 Confidence Scoring**: Provides confidence percentages for predictions
- **⚡ High Performance**: Optimized for speed with batch processing and caching
- **🐳 Docker Ready**: Easy deployment with Docker and Docker Compose

## 🚀 Quick Start

### Option 1: Automated Installation

```bash
# Clone the repository
git clone https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool.git
cd Multilingual-Sentiment-Analysis-Tool

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh

# Activate virtual environment
source venv/bin/activate

# Run demo
python run_demo.py
```

### Option 2: Manual Installation

```bash
# Clone and setup
git clone https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool.git
cd Multilingual-Sentiment-Analysis-Tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories and config
mkdir -p data results models logs
cp .env.example .env
```

### Option 3: Docker

```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Or build and run manually
docker build -t sentiment-analysis .
docker run -p 8000:8000 sentiment-analysis
```

## 🎯 Usage

### Web Interface
```bash
# Start the web interface
streamlit run app/frontend/streamlit_app.py
# Open http://localhost:8501
```

**🌐 Or try the live demo**: https://multilingual-sentiment-analysis.streamlit.app/

### API Server
```bash
# Start the API server
uvicorn app.api.main:app --reload
# API docs at http://localhost:8000/docs
```

### Python API
```python
from app.core.multilingual_analyzer import MultilingualAnalyzer

# Initialize analyzer
analyzer = MultilingualAnalyzer()

# Analyze single text
result = analyzer.analyze_text("I love this product!")
print(f"Sentiment: {result.sentiment} ({result.sentiment_confidence:.2f})")

# Batch analysis
results = analyzer.analyze_batch([
    "Great product!",
    "Terrible service.",
    "It's okay."
])
```

### Command Line Tools
```bash
# Run demo with examples
python run_demo.py

# Using Makefile
make run-api      # Start API server
make run-frontend # Start web interface
make test         # Run tests
make lint         # Code linting
```

## 📁 Project Structure

```
Multilingual-Sentiment-Analysis-Tool/
├── app/
│   ├── core/                    # Core analysis modules
│   │   ├── sentiment_analyzer.py   # Main sentiment analysis engine
│   │   ├── language_detector.py    # Language detection
│   │   ├── translator.py           # Translation service
│   │   ├── preprocessor.py         # Text preprocessing
│   │   └── multilingual_analyzer.py # Main orchestrator
│   ├── api/                     # FastAPI backend
│   │   ├── main.py                 # FastAPI application
│   │   ├── routes/                 # API routes
│   │   └── models/                 # Request/response models
│   ├── frontend/                # Streamlit web interface
│   │   └── streamlit_app.py
│   └── utils/                   # Utilities
│       ├── config.py               # Configuration management
│       └── batch_processor.py      # Batch processing
├── tests/                       # Test suite
├── docs/                        # Documentation
├── data/                        # Sample data
├── scripts/                     # Installation scripts
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Docker configuration
├── Makefile                     # Development commands
└── requirements.txt             # Python dependencies
```

## 🌐 Supported Languages

| Language | Code | Native Name | Status |
|----------|------|-------------|--------|
| English | en | English | ✅ Supported |
| Spanish | es | Español | ✅ Supported |
| French | fr | Français | ✅ Supported |
| German | de | Deutsch | ✅ Supported |
| Chinese | zh | 中文 | ✅ Supported |
| Portuguese | pt | Português | ✅ Supported |
| Italian | it | Italiano | ✅ Supported |
| Russian | ru | Русский | ✅ Supported |
| Japanese | ja | 日本語 | ✅ Supported |
| Arabic | ar | العربية | ✅ Supported |

*Other languages are automatically translated to supported languages for analysis.*

## 🔌 API Endpoints

### Sentiment Analysis
- `POST /analyze/` - Analyze single text
- `POST /analyze/batch` - Batch analysis

### Language Detection
- `POST /language/detect` - Detect language of text
- `POST /language/detect/batch` - Batch language detection
- `GET /language/supported` - List supported languages

### System
- `GET /health` - Health check and system status
- `GET /` - Service information
- `GET /docs` - Interactive API documentation

## 📊 Performance Benchmarks

| Operation | Texts | Avg Time | Throughput |
|-----------|-------|----------|------------|
| Single Analysis | 1 | ~200ms | 5 texts/sec |
| Batch Processing | 100 | ~15s | 6.7 texts/sec |
| Language Detection | 1 | ~10ms | 100 texts/sec |
| Translation | 1 | ~300ms | 3.3 texts/sec |

*Benchmarks on CPU. GPU acceleration provides 3-5x speedup.*

## 🧪 Testing

```bash
# Run all tests
make test

# Run fast tests only
make test-fast

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Integration tests only
```

## 📚 Documentation

- [API Documentation](docs/API_DOCUMENTATION.md) - Complete API reference
- [Usage Examples](docs/USAGE_EXAMPLES.md) - Code examples and tutorials
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the ML models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the web interface
- [LangDetect](https://github.com/Mimino666/langdetect) for language detection

## 🌐 Links

- 🚀 **Live Demo**: https://multilingual-sentiment-analysis.streamlit.app/
- 📖 **Repository**: https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool
- 📚 **Documentation**: [API Docs](docs/API_DOCUMENTATION.md) | [Usage Examples](docs/USAGE_EXAMPLES.md) | [Deployment Guide](docs/DEPLOYMENT.md)

## 📞 Support

- 📧 Email: midlajmuhammed@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool/discussions)
