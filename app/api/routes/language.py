"""Language detection API routes."""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends

from ..models import (
    LanguageDetectionRequest,
    BatchLanguageDetectionRequest,
    LanguageDetectionResponse,
    BatchLanguageDetectionResponse,
    LanguageInfo,
    SupportedLanguagesResponse
)
from ...core.language_detector import LanguageDetector, LanguageResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/language", tags=["language"])

# Global language detector instance
language_detector = LanguageDetector()


def get_language_detector() -> LanguageDetector:
    """Dependency to get the language detector instance."""
    return language_detector


def convert_language_result(result: LanguageResult, text: str, processing_time: float) -> LanguageDetectionResponse:
    """Convert internal LanguageResult to API response format."""
    
    language_info = LanguageInfo(
        code=result.language,
        name=language_detector.get_language_name(result.language),
        confidence=result.confidence,
        is_supported=result.is_supported,
        alternatives=result.alternatives
    )
    
    return LanguageDetectionResponse(
        text=text,
        language=language_info,
        processing_time=processing_time,
        timestamp=datetime.utcnow().isoformat()
    )


@router.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(
    request: LanguageDetectionRequest,
    detector: LanguageDetector = Depends(get_language_detector)
) -> LanguageDetectionResponse:
    """Detect language of input text.
    
    Args:
        request: Language detection request
        detector: Language detector instance
        
    Returns:
        Language detection results
    """
    try:
        logger.info(f"Detecting language for text: {request.text[:100]}...")
        
        import time
        start_time = time.time()
        
        # Perform language detection
        result = detector.detect_language(request.text)
        
        processing_time = time.time() - start_time
        
        # Convert to API response format
        response = convert_language_result(result, request.text, processing_time)
        
        logger.info(f"Language detected: {result.language} ({result.confidence:.3f})")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during language detection")


@router.post("/detect/batch", response_model=BatchLanguageDetectionResponse)
async def detect_languages_batch(
    request: BatchLanguageDetectionRequest,
    detector: LanguageDetector = Depends(get_language_detector)
) -> BatchLanguageDetectionResponse:
    """Detect languages for multiple texts in batch.
    
    Args:
        request: Batch language detection request
        detector: Language detector instance
        
    Returns:
        Batch language detection results
    """
    try:
        logger.info(f"Starting batch language detection for {len(request.texts)} texts")
        
        import time
        start_time = time.time()
        
        # Perform batch language detection
        results = detector.detect_batch(request.texts)
        
        total_processing_time = time.time() - start_time
        avg_processing_time = total_processing_time / len(request.texts)
        
        # Convert to API response format
        api_results = [
            convert_language_result(result, text, avg_processing_time)
            for result, text in zip(results, request.texts)
        ]
        
        # Calculate summary statistics
        languages = [result.language for result in results]
        confidences = [result.confidence for result in results]
        supported_count = sum(1 for result in results if result.is_supported)
        
        summary = {
            "total_texts": len(request.texts),
            "language_distribution": {lang: languages.count(lang) for lang in set(languages)},
            "supported_languages": supported_count,
            "unsupported_languages": len(request.texts) - supported_count,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0
        }
        
        response = BatchLanguageDetectionResponse(
            results=api_results,
            summary=summary,
            total_processing_time=total_processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Batch language detection completed: {len(results)} texts processed")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch language detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch language detection")


@router.get("/supported", response_model=SupportedLanguagesResponse)
async def get_supported_languages(
    detector: LanguageDetector = Depends(get_language_detector)
) -> SupportedLanguagesResponse:
    """Get list of supported languages for sentiment analysis.
    
    Args:
        detector: Language detector instance
        
    Returns:
        Supported languages information
    """
    try:
        supported_languages = detector.get_supported_languages()
        
        response = SupportedLanguagesResponse(
            languages=supported_languages,
            count=len(supported_languages),
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
