# Usage Examples

This document provides comprehensive examples of how to use the Multilingual Sentiment Analysis Tool.

**🚀 Live Demo**: https://multilingual-sentiment-analysis.streamlit.app/
**📖 Repository**: https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool

## Table of Contents

1. [Python API Usage](#python-api-usage)
2. [REST API Usage](#rest-api-usage)
3. [Web Interface Usage](#web-interface-usage)
4. [Batch Processing](#batch-processing)
5. [Advanced Configuration](#advanced-configuration)

## Python API Usage

### Basic Sentiment Analysis

```python
from app.core.multilingual_analyzer import MultilingualAnalyzer

# Initialize analyzer
analyzer = MultilingualAnalyzer()

# Analyze single text
result = analyzer.analyze_text("I love this product!")

print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.sentiment_confidence:.2f}")
print(f"Language: {result.detected_language}")
print(f"Scores: {result.sentiment_scores}")
```

### Multilingual Analysis

```python
# Analyze texts in different languages
texts = [
    "I love this product!",           # English
    "Me encanta este producto!",      # Spanish  
    "J'adore ce produit!",           # French
    "Ich liebe dieses Produkt!",     # German
    "我喜欢这个产品！"                 # Chinese
]

for text in texts:
    result = analyzer.analyze_text(text)
    print(f"Text: {text}")
    print(f"Language: {result.detected_language} ({result.language_confidence:.2f})")
    print(f"Sentiment: {result.sentiment} ({result.sentiment_confidence:.2f})")
    print(f"Translation needed: {result.translation_needed}")
    print("---")
```

### Batch Processing

```python
# Analyze multiple texts efficiently
texts = [
    "Great product, highly recommend!",
    "Terrible quality, waste of money.",
    "It's okay, nothing special.",
    "Amazing customer service!",
    "Disappointed with the purchase."
]

results = analyzer.analyze_batch(texts)

# Print summary
sentiments = [r.sentiment for r in results]
print(f"Positive: {sentiments.count('positive')}")
print(f"Negative: {sentiments.count('negative')}")
print(f"Neutral: {sentiments.count('neutral')}")
```

### Language Detection Only

```python
from app.core.language_detector import LanguageDetector

detector = LanguageDetector()

# Detect language
result = detector.detect_language("Bonjour, comment allez-vous?")
print(f"Language: {result.language}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Supported: {result.is_supported}")
print(f"Alternatives: {result.alternatives}")
```

### Translation Service

```python
from app.core.translator import TranslationService

translator = TranslationService()

# Translate text
result = translator.translate_text(
    "Hola, ¿cómo estás?", 
    source_language="es",
    target_language="en"
)

print(f"Original: {result.original_text}")
print(f"Translated: {result.translated_text}")
print(f"Confidence: {result.confidence:.2f}")
```

## REST API Usage

### Using curl

#### Single Text Analysis

```bash
curl -X POST "http://localhost:8000/analyze/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this product!",
    "enable_translation": true,
    "enable_preprocessing": true
  }'
```

#### Batch Analysis

```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product!",
      "Terrible service.",
      "It'\''s okay."
    ],
    "enable_translation": true,
    "enable_preprocessing": true
  }'
```

#### Language Detection

```bash
curl -X POST "http://localhost:8000/language/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour le monde"
  }'
```

### Using Python requests

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Single text analysis
response = requests.post(
    f"{BASE_URL}/analyze/",
    json={
        "text": "I absolutely love this product!",
        "enable_translation": True,
        "enable_preprocessing": True
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")

# Batch analysis
texts = [
    "Excellent quality and fast delivery!",
    "Poor customer service, very disappointed.",
    "The product is average, nothing special."
]

response = requests.post(
    f"{BASE_URL}/analyze/batch",
    json={
        "texts": texts,
        "enable_translation": True,
        "enable_preprocessing": True
    }
)

batch_result = response.json()
print(f"Processed {len(batch_result['results'])} texts")
print(f"Summary: {batch_result['summary']}")
```

### Using JavaScript/Node.js

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function analyzeSentiment(text) {
  try {
    const response = await axios.post(`${BASE_URL}/analyze/`, {
      text: text,
      enable_translation: true,
      enable_preprocessing: true
    });
    
    const result = response.data;
    console.log(`Sentiment: ${result.sentiment}`);
    console.log(`Confidence: ${result.confidence}`);
    console.log(`Language: ${result.language.name}`);
    
    return result;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

// Usage
analyzeSentiment("I love this product!");
```

## Web Interface Usage

### Starting the Web Interface

```bash
# Start the Streamlit app
streamlit run app/frontend/streamlit_app.py
```

### Features

1. **Single Text Analysis**
   - Enter text in the text area
   - Configure translation and preprocessing options
   - Click "Analyze" to get results
   - View detailed scores and information

2. **Batch Analysis**
   - Upload CSV file with 'text' column
   - Or enter multiple texts manually (one per line)
   - Download results as CSV
   - View analytics dashboard

3. **Analytics Dashboard**
   - Sentiment distribution charts
   - Confidence score histograms
   - Language distribution
   - Processing statistics

## Batch Processing

### CSV File Processing

```python
from app.utils.batch_processor import BatchProcessor
from app.core.multilingual_analyzer import MultilingualAnalyzer

# Initialize components
analyzer = MultilingualAnalyzer()
processor = BatchProcessor(analyzer, batch_size=32, max_workers=4)

# Process CSV file
results = processor.process_csv_file(
    file_path="data/reviews.csv",
    text_column="review_text",
    output_path="results/sentiment_results.csv",
    include_metadata=True
)

print(f"Processed {len(results)} texts")
```

### Progress Tracking

```python
def progress_callback(processed, total):
    percentage = (processed / total) * 100
    print(f"Progress: {processed}/{total} ({percentage:.1f}%)")

processor = BatchProcessor(
    analyzer, 
    progress_callback=progress_callback
)

results = processor.process_texts(texts)
```

### Large Dataset Processing

```python
import pandas as pd

# Read large CSV in chunks
chunk_size = 1000
results = []

for chunk in pd.read_csv("large_dataset.csv", chunksize=chunk_size):
    texts = chunk['text'].tolist()
    chunk_results = analyzer.analyze_batch(texts)
    results.extend(chunk_results)
    
    print(f"Processed {len(results)} texts so far...")

# Save final results
processor.save_results(results, "final_results.json")
```

## Advanced Configuration

### Custom Model Configuration

```python
# Use custom model
analyzer = MultilingualAnalyzer(
    model_name="nlptown/bert-base-multilingual-uncased-sentiment",
    cache_dir="./custom_models",
    device="cuda",  # Use GPU if available
    language_confidence_threshold=0.8,
    enable_translation=True,
    enable_preprocessing=True
)
```

### Environment Configuration

```bash
# .env file
DEBUG=False
LOG_LEVEL=INFO
DEFAULT_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment
CACHE_DIR=./models
MAX_LENGTH=512
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=10
GOOGLE_TRANSLATE_API_KEY=your_api_key_here
REDIS_URL=redis://localhost:6379
```

### Custom Preprocessing

```python
from app.core.preprocessor import TextPreprocessor

# Custom preprocessor
preprocessor = TextPreprocessor(
    remove_urls=True,
    remove_emails=True,
    remove_mentions=True,
    remove_hashtags=False,  # Keep hashtags
    normalize_unicode=True,
    min_length=5,
    max_length=1000
)

# Use with analyzer
analyzer.preprocessor = preprocessor
```

### Performance Optimization

```python
# For high-throughput scenarios
analyzer = MultilingualAnalyzer(
    device="cuda",  # Use GPU
    enable_preprocessing=False,  # Skip preprocessing for speed
    enable_translation=False     # Skip translation for speed
)

# Batch processing with optimal settings
processor = BatchProcessor(
    analyzer,
    batch_size=64,      # Larger batches
    max_workers=8       # More parallel workers
)
```

### Error Handling

```python
try:
    result = analyzer.analyze_text(text)
    if result.sentiment_confidence < 0.5:
        print("Warning: Low confidence result")
except Exception as e:
    print(f"Analysis failed: {e}")
    # Handle error appropriately
```

### Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monitor performance
import time

start_time = time.time()
results = analyzer.analyze_batch(texts)
processing_time = time.time() - start_time

logger.info(f"Processed {len(texts)} texts in {processing_time:.2f}s")
logger.info(f"Average time per text: {processing_time/len(texts):.3f}s")
```
