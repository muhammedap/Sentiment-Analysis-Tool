#!/bin/bash

# Multilingual Sentiment Analysis Tool - Installation Script

set -e  # Exit on any error

echo "🌍 Installing Multilingual Sentiment Analysis Tool"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data results models logs

# Copy environment configuration
if [ ! -f .env ]; then
    echo "⚙️  Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Download sample data (if not exists)
if [ ! -f data/sample_reviews.csv ]; then
    echo "📄 Sample data already exists"
fi

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. 🌐 Try the live demo: https://multilingual-sentiment-analysis.streamlit.app/"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Edit .env file with your configuration"
echo "4. Run the demo: python run_demo.py"
echo "5. Start the API server: make run-api"
echo "6. Start the web interface: make run-frontend"
echo ""
echo "🚀 Live Demo: https://multilingual-sentiment-analysis.streamlit.app/"
echo "📖 Repository: https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool"
echo "For more information, see README.md and docs/"
