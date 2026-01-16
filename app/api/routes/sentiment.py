"""Sentiment analysis API routes."""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    TextAnalysisRequest,
    BatchAnalysisRequest,
    AnalysisResponse,
    BatchAnalysisResponse,
    SentimentScores,
    LanguageInfo,
    TranslationInfo,
    PreprocessingInfo,
    ErrorResponse
)
from ...core.multilingual_analyzer import MultilingualAnalyzer, AnalysisResult
from ...utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["sentiment"])

# Global analyzer instance (will be initialized on startup)
analyzer: MultilingualAnalyzer = None


def get_analyzer() -> MultilingualAnalyzer:
    """Dependency to get the analyzer instance."""
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Sentiment analyzer not initialized"
        )
    return analyzer


def convert_analysis_result(result: AnalysisResult) -> AnalysisResponse:
    """Convert internal AnalysisResult to API response format."""
    
    # Language info
    language_info = LanguageInfo(
        code=result.detected_language,
        name=analyzer.language_detector.get_language_name(result.detected_language),
        confidence=result.language_confidence,
        is_supported=analyzer.language_detector.is_supported(result.detected_language),
        alternatives=result.language_alternatives
    )
    
    # Translation info
    translation_info = TranslationInfo(
        needed=result.translation_needed,
        source_language=result.detected_language if result.translation_needed else None,
        target_language="en" if result.translation_needed else None,  # Assuming English target
        translated_text=result.translated_text,
        confidence=result.translation_confidence,
        cached=result.cached_translation
    )
    
    # Preprocessing info
    preprocessing_info = PreprocessingInfo(
        enabled=analyzer.enable_preprocessing,
        original_length=result.preprocessing_stats.get("original_length") if result.preprocessing_stats else None,
        processed_length=result.preprocessing_stats.get("processed_length") if result.preprocessing_stats else None,
        length_reduction=result.preprocessing_stats.get("length_reduction") if result.preprocessing_stats else None,
        reduction_percentage=result.preprocessing_stats.get("reduction_percentage") if result.preprocessing_stats else None
    )
    
    # Sentiment scores
    sentiment_scores = SentimentScores(
        positive=result.sentiment_scores.get("positive", 0.0),
        negative=result.sentiment_scores.get("negative", 0.0),
        neutral=result.sentiment_scores.get("neutral", 0.0)
    )
    
    return AnalysisResponse(
        original_text=result.original_text,
        sentiment=result.sentiment,
        confidence=result.sentiment_confidence,
        scores=sentiment_scores,
        language=language_info,
        translation=translation_info,
        preprocessing=preprocessing_info,
        processing_time=result.processing_time,
        model_used=result.model_used,
        timestamp=datetime.utcnow().isoformat()
    )


@router.post("/", response_model=AnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    analyzer: MultilingualAnalyzer = Depends(get_analyzer)
) -> AnalysisResponse:
    """Analyze sentiment of a single text.
    
    Args:
        request: Text analysis request
        analyzer: Sentiment analyzer instance
        
    Returns:
        Analysis results
    """
    try:
        logger.info(f"Analyzing text: {request.text[:100]}...")
        
        # Configure analyzer based on request
        analyzer.enable_translation = request.enable_translation
        analyzer.enable_preprocessing = request.enable_preprocessing
        
        # Perform analysis
        result = analyzer.analyze_text(request.text)
        
        # Convert to API response format
        response = convert_analysis_result(result)
        
        logger.info(f"Analysis completed: {result.sentiment} ({result.sentiment_confidence:.3f})")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")


@router.post("/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    analyzer: MultilingualAnalyzer = Depends(get_analyzer)
) -> BatchAnalysisResponse:
    """Analyze sentiment of multiple texts in batch.
    
    Args:
        request: Batch analysis request
        background_tasks: Background tasks for cleanup
        analyzer: Sentiment analyzer instance
        
    Returns:
        Batch analysis results
    """
    try:
        logger.info(f"Starting batch analysis of {len(request.texts)} texts")
        
        # Configure analyzer based on request
        analyzer.enable_translation = request.enable_translation
        analyzer.enable_preprocessing = request.enable_preprocessing
        
        # Perform batch analysis
        results = analyzer.analyze_batch(request.texts)
        
        # Convert to API response format
        api_results = [convert_analysis_result(result) for result in results]
        
        # Calculate summary statistics
        sentiments = [result.sentiment for result in results]
        languages = [result.detected_language for result in results]
        translations_needed = sum(1 for result in results if result.translation_needed)
        
        summary = {
            "total_texts": len(request.texts),
            "sentiment_distribution": {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            },
            "language_distribution": {lang: languages.count(lang) for lang in set(languages)},
            "translations_needed": translations_needed,
            "average_confidence": sum(result.sentiment_confidence for result in results) / len(results),
            "average_processing_time": sum(result.processing_time for result in results) / len(results)
        }
        
        total_processing_time = sum(result.processing_time for result in results)
        
        response = BatchAnalysisResponse(
            results=api_results,
            summary=summary,
            total_processing_time=total_processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Batch analysis completed: {len(results)} texts processed")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch analysis")


# Initialize analyzer (called during app startup)
def initialize_analyzer():
    """Initialize the global analyzer instance."""
    global analyzer
    try:
        logger.info("Initializing sentiment analyzer...")
        analyzer = MultilingualAnalyzer()
        logger.info("Sentiment analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise
