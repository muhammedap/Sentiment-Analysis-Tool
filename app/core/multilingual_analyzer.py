"""Main multilingual sentiment analyzer that orchestrates all components."""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .sentiment_analyzer import MultilingualSentimentAnalyzer, SentimentResult
from .language_detector import LanguageDetector, LanguageResult
from .translator import TranslationService, TranslationResult
from .preprocessor import TextPreprocessor
from ..utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result including all processing steps."""
    # Input
    original_text: str

    # Language detection
    detected_language: str
    language_confidence: float
    language_alternatives: List[Tuple[str, float]]

    # Translation (if needed)
    translation_needed: bool

    # Preprocessing
    preprocessed_text: str

    # Sentiment analysis
    sentiment: str
    sentiment_confidence: float
    sentiment_scores: Dict[str, float]

    # Metadata
    processing_time: float
    model_used: str

    # Optional fields with defaults
    translated_text: Optional[str] = None
    translation_confidence: Optional[float] = None
    preprocessing_stats: Optional[Dict] = None
    cached_translation: bool = False


class MultilingualAnalyzer:
    """Main analyzer that orchestrates all components."""
    
    def __init__(self,
                 model_name: str = None,
                 cache_dir: str = None,
                 device: str = None,
                 language_confidence_threshold: float = 0.7,
                 enable_translation: bool = True,
                 enable_preprocessing: bool = True):
        """Initialize the multilingual analyzer.
        
        Args:
            model_name: Sentiment analysis model name
            cache_dir: Model cache directory
            device: Device for inference
            language_confidence_threshold: Minimum confidence for language detection
            enable_translation: Enable translation for unsupported languages
            enable_preprocessing: Enable text preprocessing
        """
        self.enable_translation = enable_translation
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize components
        logger.info("Initializing multilingual analyzer components...")
        
        # Language detector
        self.language_detector = LanguageDetector(
            confidence_threshold=language_confidence_threshold
        )
        
        # Translation service
        if self.enable_translation:
            self.translation_service = TranslationService()
        else:
            self.translation_service = None
        
        # Text preprocessor
        if self.enable_preprocessing:
            self.preprocessor = TextPreprocessor(
                max_length=settings.max_length * 4  # Rough character estimate
            )
        else:
            self.preprocessor = None
        
        # Sentiment analyzer (loaded last as it's most resource-intensive)
        self.sentiment_analyzer = MultilingualSentimentAnalyzer(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )
        
        logger.info("Multilingual analyzer initialized successfully")
    
    def analyze_text(self, text: str) -> AnalysisResult:
        """Analyze sentiment of a single text with full pipeline.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complete analysis result
        """
        start_time = time.time()
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        original_text = text
        
        # Step 1: Language detection
        logger.debug("Detecting language...")
        lang_result = self.language_detector.detect_language(text)
        
        # Step 2: Translation (if needed)
        translation_needed = False
        translated_text = None
        translation_confidence = None
        cached_translation = False
        
        if (self.enable_translation and 
            self.translation_service and 
            not lang_result.is_supported):
            
            logger.debug(f"Translating from {lang_result.language} to supported language...")
            translation_needed = True
            
            translation_result = self.translation_service.translate_text(
                text, lang_result.language
            )
            
            translated_text = translation_result.translated_text
            translation_confidence = translation_result.confidence
            cached_translation = translation_result.cached
            
            # Use translated text for further processing
            analysis_text = translated_text
        else:
            analysis_text = text
        
        # Step 3: Preprocessing
        preprocessing_stats = None
        if self.enable_preprocessing and self.preprocessor:
            logger.debug("Preprocessing text...")
            preprocessed_text = self.preprocessor.preprocess_text(analysis_text)
            preprocessing_stats = self.preprocessor.get_preprocessing_stats(
                analysis_text, preprocessed_text
            )
            
            if not preprocessed_text:
                logger.warning("Text became empty after preprocessing")
                preprocessed_text = analysis_text  # Fallback to original
        else:
            preprocessed_text = analysis_text
        
        # Step 4: Sentiment analysis
        logger.debug("Analyzing sentiment...")
        sentiment_result = self.sentiment_analyzer.analyze_text(
            preprocessed_text,
            language=lang_result.language,
            translated_text=translated_text
        )
        
        # Compile final result
        processing_time = time.time() - start_time
        
        result = AnalysisResult(
            original_text=original_text,
            detected_language=lang_result.language,
            language_confidence=lang_result.confidence,
            language_alternatives=lang_result.alternatives,
            translation_needed=translation_needed,
            translated_text=translated_text,
            translation_confidence=translation_confidence,
            preprocessed_text=preprocessed_text,
            preprocessing_stats=preprocessing_stats,
            sentiment=sentiment_result.sentiment,
            sentiment_confidence=sentiment_result.confidence,
            sentiment_scores=sentiment_result.scores,
            processing_time=processing_time,
            model_used=self.sentiment_analyzer.model_name,
            cached_translation=cached_translation
        )
        
        logger.debug(f"Analysis completed in {processing_time:.3f}s")
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[AnalysisResult]:
        """Analyze multiple texts with optimized batch processing.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        if not texts:
            return []
        
        start_time = time.time()
        logger.info(f"Starting batch analysis of {len(texts)} texts")
        
        # Step 1: Batch language detection
        lang_results = self.language_detector.detect_batch(texts)
        
        # Step 2: Batch translation (if needed)
        translation_results = []
        texts_for_analysis = []
        
        if self.enable_translation and self.translation_service:
            for text, lang_result in zip(texts, lang_results):
                if not lang_result.is_supported:
                    translation_result = self.translation_service.translate_text(
                        text, lang_result.language
                    )
                    translation_results.append(translation_result)
                    texts_for_analysis.append(translation_result.translated_text)
                else:
                    translation_results.append(None)
                    texts_for_analysis.append(text)
        else:
            translation_results = [None] * len(texts)
            texts_for_analysis = texts.copy()
        
        # Step 3: Batch preprocessing
        if self.enable_preprocessing and self.preprocessor:
            preprocessed_texts = self.preprocessor.preprocess_batch(texts_for_analysis)
        else:
            preprocessed_texts = texts_for_analysis.copy()
        
        # Step 4: Batch sentiment analysis
        languages = [result.language for result in lang_results]
        translated_texts = [
            result.translated_text if result else None 
            for result in translation_results
        ]
        
        sentiment_results = self.sentiment_analyzer.analyze_batch(
            preprocessed_texts,
            languages=languages,
            translated_texts=translated_texts
        )
        
        # Compile final results
        final_results = []
        for i, (text, lang_result, translation_result, preprocessed_text, sentiment_result) in enumerate(
            zip(texts, lang_results, translation_results, preprocessed_texts, sentiment_results)
        ):
            processing_time = (time.time() - start_time) / len(texts)  # Average time
            
            result = AnalysisResult(
                original_text=text,
                detected_language=lang_result.language,
                language_confidence=lang_result.confidence,
                language_alternatives=lang_result.alternatives,
                translation_needed=translation_result is not None,
                translated_text=translation_result.translated_text if translation_result else None,
                translation_confidence=translation_result.confidence if translation_result else None,
                preprocessed_text=preprocessed_text,
                preprocessing_stats=None,  # Skip for batch processing
                sentiment=sentiment_result.sentiment,
                sentiment_confidence=sentiment_result.confidence,
                sentiment_scores=sentiment_result.scores,
                processing_time=processing_time,
                model_used=self.sentiment_analyzer.model_name,
                cached_translation=translation_result.cached if translation_result else False
            )
            final_results.append(result)
        
        total_time = time.time() - start_time
        logger.info(f"Batch analysis completed in {total_time:.3f}s")
        
        return final_results
    
    def get_system_info(self) -> Dict:
        """Get system information and component status.
        
        Returns:
            System information dictionary
        """
        info = {
            "sentiment_analyzer": self.sentiment_analyzer.get_model_info(),
            "supported_languages": self.language_detector.get_supported_languages(),
            "translation_enabled": self.enable_translation,
            "preprocessing_enabled": self.enable_preprocessing,
        }
        
        if self.translation_service:
            info["translation_cache"] = self.translation_service.get_cache_stats()
        
        return info
