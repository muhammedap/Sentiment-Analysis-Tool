# API Documentation

**🚀 Live Demo**: https://multilingual-sentiment-analysis.streamlit.app/
**📖 Repository**: https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool

## Overview

The Multilingual Sentiment Analysis API provides endpoints for analyzing sentiment in text across multiple languages. The API supports automatic language detection, translation fallback for unsupported languages, and batch processing capabilities.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production deployment, consider implementing API key authentication.

## Endpoints

### Health Check

#### GET /health

Returns the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "supported_languages": ["en", "es", "fr", "de", "zh", "pt", "it", "ru", "ja", "ar"],
  "system_info": {
    "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "device": "cpu",
    "cpu_percent": 15.2,
    "memory_percent": 45.8
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET /

Returns basic service information.

**Response:**
```json
{
  "service": "Multilingual Sentiment Analysis Tool",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

### Sentiment Analysis

#### POST /analyze/

Analyze sentiment of a single text.

**Request Body:**
```json
{
  "text": "I love this product!",
  "language": "en",
  "enable_translation": true,
  "enable_preprocessing": true
}
```

**Parameters:**
- `text` (string, required): Text to analyze (1-10000 characters)
- `language` (string, optional): Language hint (ISO 639-1 code)
- `enable_translation` (boolean, optional): Enable translation for unsupported languages (default: true)
- `enable_preprocessing` (boolean, optional): Enable text preprocessing (default: true)

**Response:**
```json
{
  "original_text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.85,
  "scores": {
    "positive": 0.85,
    "negative": 0.10,
    "neutral": 0.05
  },
  "language": {
    "code": "en",
    "name": "English",
    "confidence": 0.95,
    "is_supported": true,
    "alternatives": [["es", 0.03], ["fr", 0.02]]
  },
  "translation": {
    "needed": false,
    "source_language": null,
    "target_language": null,
    "translated_text": null,
    "confidence": null,
    "cached": false
  },
  "preprocessing": {
    "enabled": true,
    "original_length": 18,
    "processed_length": 18,
    "length_reduction": 0,
    "reduction_percentage": 0.0
  },
  "processing_time": 0.123,
  "model_used": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /analyze/batch

Analyze sentiment of multiple texts in batch.

**Request Body:**
```json
{
  "texts": [
    "I love this product!",
    "This is terrible.",
    "The weather is okay."
  ],
  "languages": ["en", "en", "en"],
  "enable_translation": true,
  "enable_preprocessing": true
}
```

**Parameters:**
- `texts` (array, required): List of texts to analyze (1-100 items)
- `languages` (array, optional): Language hints for each text
- `enable_translation` (boolean, optional): Enable translation (default: true)
- `enable_preprocessing` (boolean, optional): Enable preprocessing (default: true)

**Response:**
```json
{
  "results": [
    {
      "original_text": "I love this product!",
      "sentiment": "positive",
      "confidence": 0.85,
      // ... (same structure as single analysis)
    }
  ],
  "summary": {
    "total_texts": 3,
    "sentiment_distribution": {
      "positive": 1,
      "negative": 1,
      "neutral": 1
    },
    "language_distribution": {
      "en": 3
    },
    "translations_needed": 0,
    "average_confidence": 0.78,
    "average_processing_time": 0.045
  },
  "total_processing_time": 0.135,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Language Detection

#### POST /language/detect

Detect language of input text.

**Request Body:**
```json
{
  "text": "Bonjour le monde"
}
```

**Response:**
```json
{
  "text": "Bonjour le monde",
  "language": {
    "code": "fr",
    "name": "French",
    "confidence": 0.92,
    "is_supported": true,
    "alternatives": [["en", 0.05], ["es", 0.03]]
  },
  "processing_time": 0.012,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /language/detect/batch

Detect languages for multiple texts.

**Request Body:**
```json
{
  "texts": [
    "Hello world",
    "Hola mundo",
    "Bonjour le monde"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Hello world",
      "language": {
        "code": "en",
        "name": "English",
        "confidence": 0.95,
        "is_supported": true,
        "alternatives": []
      },
      "processing_time": 0.008,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "summary": {
    "total_texts": 3,
    "language_distribution": {
      "en": 1,
      "es": 1,
      "fr": 1
    },
    "supported_languages": 3,
    "unsupported_languages": 0,
    "average_confidence": 0.91
  },
  "total_processing_time": 0.024,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET /language/supported

Get list of supported languages.

**Response:**
```json
{
  "languages": {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ja": "Japanese",
    "ar": "Arabic"
  },
  "count": 10,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": {
    "errors": [
      {
        "loc": ["body", "text"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc123"
}
```

### Error Types

- `validation_error` (422): Request validation failed
- `http_error` (4xx): HTTP client error
- `internal_error` (500): Internal server error

## Rate Limiting

Currently, no rate limiting is implemented. For production deployment, consider implementing rate limiting based on your requirements.

## Supported Languages

The API supports sentiment analysis in the following languages:

| Code | Language | Native Name |
|------|----------|-------------|
| en   | English  | English     |
| es   | Spanish  | Español     |
| fr   | French   | Français    |
| de   | German   | Deutsch     |
| zh   | Chinese  | 中文        |
| pt   | Portuguese | Português |
| it   | Italian  | Italiano    |
| ru   | Russian  | Русский     |
| ja   | Japanese | 日本語      |
| ar   | Arabic   | العربية     |

For unsupported languages, the API can automatically translate text to a supported language before analysis (if translation is enabled).

## Performance Considerations

- Single text analysis: ~100-500ms depending on text length and model
- Batch processing: More efficient for multiple texts
- Translation adds ~200-500ms per text
- Preprocessing adds ~10-50ms per text
- GPU acceleration significantly improves performance

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI Schema: `/openapi.json`
