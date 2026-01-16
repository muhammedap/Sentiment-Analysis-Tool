"""Core sentiment analysis engine using multilingual transformer models."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
from transformers.pipelines.base import KeyDataset

from ..utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]
    processing_time: float
    language: Optional[str] = None
    translated_text: Optional[str] = None


class MultilingualSentimentAnalyzer:
    """Multilingual sentiment analysis engine."""

    SENTIMENT_LABELS = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        # Star ratings mapping (for nlptown model)
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive"
    }
    
    def __init__(
        self,
        model_name: str = None,
        cache_dir: str = None,
        device: str = None
    ):
        """Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the model to use
            cache_dir: Directory to cache models
            device: Device to run inference on
        """
        self.model_name = model_name or settings.default_model
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {self.model_name}")
        
        # Initialize model and tokenizer
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the sentiment analysis model and tokenizer."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                top_k=None  # Return all scores (replaces deprecated return_all_scores)
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _normalize_sentiment_labels(self, results: List[Dict]) -> Dict[str, float]:
        """Normalize sentiment labels to standard format."""
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for result in results:
            label = result["label"]
            score = result["score"]

            # Map model-specific labels to standard labels
            if label in self.SENTIMENT_LABELS:
                standard_label = self.SENTIMENT_LABELS[label]
            else:
                # Handle different model label formats
                standard_label = label.lower()
                if "pos" in standard_label or "5" in standard_label or "4" in standard_label:
                    standard_label = "positive"
                elif "neg" in standard_label or "1" in standard_label or "2" in standard_label:
                    standard_label = "negative"
                elif "neu" in standard_label or "3" in standard_label:
                    standard_label = "neutral"
                else:
                    # Default fallback
                    standard_label = "neutral"

            # Accumulate scores for the same sentiment
            if standard_label in scores:
                scores[standard_label] += score
            else:
                scores[standard_label] = score

        return scores
    
    def analyze_text(
        self, 
        text: str, 
        language: Optional[str] = None,
        translated_text: Optional[str] = None
    ) -> SentimentResult:
        """Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            language: Detected language of the text
            translated_text: Translated version if translation was used
            
        Returns:
            SentimentResult object with analysis results
        """
        start_time = time.time()
        
        try:
            # Use translated text for analysis if available
            analysis_text = translated_text or text
            
            # Truncate text if too long
            if len(analysis_text) > settings.max_length * 4:  # Rough character estimate
                analysis_text = analysis_text[:settings.max_length * 4]
            
            # Run sentiment analysis
            results = self.pipeline(analysis_text)
            
            # Normalize scores
            scores = self._normalize_sentiment_labels(results[0])
            
            # Determine primary sentiment
            primary_sentiment = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[primary_sentiment]
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                text=text,
                sentiment=primary_sentiment,
                confidence=confidence,
                scores=scores,
                processing_time=processing_time,
                language=language,
                translated_text=translated_text
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            processing_time = time.time() - start_time
            
            # Return neutral result on error
            return SentimentResult(
                text=text,
                sentiment="neutral",
                confidence=0.0,
                scores={"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                processing_time=processing_time,
                language=language,
                translated_text=translated_text
            )
    
    def analyze_batch(
        self, 
        texts: List[str],
        languages: Optional[List[str]] = None,
        translated_texts: Optional[List[str]] = None
    ) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            languages: List of detected languages
            translated_texts: List of translated texts if translation was used
            
        Returns:
            List of SentimentResult objects
        """
        start_time = time.time()
        results = []
        
        try:
            # Prepare analysis texts
            analysis_texts = []
            for i, text in enumerate(texts):
                if translated_texts and i < len(translated_texts) and translated_texts[i]:
                    analysis_text = translated_texts[i]
                else:
                    analysis_text = text
                
                # Truncate if necessary
                if len(analysis_text) > settings.max_length * 4:
                    analysis_text = analysis_text[:settings.max_length * 4]
                    
                analysis_texts.append(analysis_text)
            
            # Run batch analysis
            batch_results = self.pipeline(analysis_texts)
            
            # Process results
            for i, (text, pipeline_result) in enumerate(zip(texts, batch_results)):
                scores = self._normalize_sentiment_labels(pipeline_result)
                primary_sentiment = max(scores.keys(), key=lambda k: scores[k])
                confidence = scores[primary_sentiment]
                
                language = languages[i] if languages and i < len(languages) else None
                translated_text = (translated_texts[i] 
                                 if translated_texts and i < len(translated_texts) 
                                 else None)
                
                result = SentimentResult(
                    text=text,
                    sentiment=primary_sentiment,
                    confidence=confidence,
                    scores=scores,
                    processing_time=(time.time() - start_time) / len(texts),  # Average time
                    language=language,
                    translated_text=translated_text
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            # Return neutral results for all texts on error
            for i, text in enumerate(texts):
                language = languages[i] if languages and i < len(languages) else None
                translated_text = (translated_texts[i] 
                                 if translated_texts and i < len(translated_texts) 
                                 else None)
                
                result = SentimentResult(
                    text=text,
                    sentiment="neutral",
                    confidence=0.0,
                    scores={"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                    processing_time=(time.time() - start_time) / len(texts),
                    language=language,
                    translated_text=translated_text
                )
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_dir": str(self.cache_dir),
            "max_length": str(settings.max_length)
        }
