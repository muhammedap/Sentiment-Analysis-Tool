"""Batch processing utilities for handling large datasets."""

import logging
import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Callable
from dataclasses import asdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..core.multilingual_analyzer import MultilingualAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processor for handling large-scale sentiment analysis."""
    
    def __init__(self, 
                 analyzer: MultilingualAnalyzer,
                 batch_size: int = 32,
                 max_workers: int = 4,
                 progress_callback: Optional[Callable[[int, int], None]] = None):
        """Initialize batch processor.
        
        Args:
            analyzer: Sentiment analyzer instance
            batch_size: Number of texts to process in each batch
            max_workers: Maximum number of worker threads
            progress_callback: Callback function for progress updates
        """
        self.analyzer = analyzer
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self._lock = threading.Lock()
        self._processed_count = 0
        
    def _update_progress(self, processed: int, total: int) -> None:
        """Update progress and call callback if provided."""
        with self._lock:
            self._processed_count = processed
            if self.progress_callback:
                self.progress_callback(processed, total)
    
    def process_csv_file(self, 
                        file_path: str, 
                        text_column: str = 'text',
                        output_path: Optional[str] = None,
                        include_metadata: bool = True) -> List[AnalysisResult]:
        """Process texts from a CSV file.
        
        Args:
            file_path: Path to input CSV file
            text_column: Name of the column containing texts
            output_path: Path to save results (optional)
            include_metadata: Include processing metadata in output
            
        Returns:
            List of analysis results
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate text column
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV file")
        
        # Extract texts
        texts = df[text_column].dropna().astype(str).tolist()
        
        if not texts:
            raise ValueError("No valid texts found in CSV file")
        
        logger.info(f"Found {len(texts)} texts to process")
        
        # Process texts
        results = self.process_texts(texts)
        
        # Save results if output path provided
        if output_path:
            self.save_results(results, output_path, include_metadata)
        
        return results
    
    def process_texts(self, texts: List[str]) -> List[AnalysisResult]:
        """Process a list of texts with batch optimization.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        if not texts:
            return []
        
        logger.info(f"Starting batch processing of {len(texts)} texts")
        start_time = time.time()
        
        # Reset progress counter
        self._processed_count = 0
        
        # Split texts into batches
        batches = [
            texts[i:i + self.batch_size] 
            for i in range(0, len(texts), self.batch_size)
        ]
        
        logger.info(f"Split into {len(batches)} batches of size {self.batch_size}")
        
        all_results = []
        
        # Process batches
        if self.max_workers == 1:
            # Single-threaded processing
            for i, batch in enumerate(batches):
                batch_results = self.analyzer.analyze_batch(batch)
                all_results.extend(batch_results)
                
                processed = (i + 1) * self.batch_size
                processed = min(processed, len(texts))
                self._update_progress(processed, len(texts))
                
                logger.debug(f"Processed batch {i+1}/{len(batches)}")
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self.analyzer.analyze_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                # Collect results as they complete
                batch_results = [None] * len(batches)
                completed_batches = 0
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        batch_results[batch_idx] = result
                        completed_batches += 1
                        
                        processed = min(completed_batches * self.batch_size, len(texts))
                        self._update_progress(processed, len(texts))
                        
                        logger.debug(f"Completed batch {batch_idx+1}/{len(batches)}")
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {e}")
                        # Create empty results for failed batch
                        batch_size = len(batches[batch_idx])
                        batch_results[batch_idx] = [
                            self._create_error_result(text, str(e)) 
                            for text in batches[batch_idx]
                        ]
                
                # Flatten results in correct order
                for batch_result in batch_results:
                    if batch_result:
                        all_results.extend(batch_result)
        
        processing_time = time.time() - start_time
        logger.info(f"Batch processing completed in {processing_time:.2f}s")
        logger.info(f"Average time per text: {processing_time/len(texts):.3f}s")
        
        return all_results
    
    def _create_error_result(self, text: str, error_message: str) -> AnalysisResult:
        """Create an error result for failed analysis."""
        from ..core.multilingual_analyzer import AnalysisResult
        
        return AnalysisResult(
            original_text=text,
            detected_language="unknown",
            language_confidence=0.0,
            language_alternatives=[],
            translation_needed=False,
            translated_text=None,
            translation_confidence=None,
            preprocessed_text=text,
            preprocessing_stats=None,
            sentiment="neutral",
            sentiment_confidence=0.0,
            sentiment_scores={"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            processing_time=0.0,
            model_used="error",
            cached_translation=False
        )
    
    def save_results(self, 
                    results: List[AnalysisResult], 
                    output_path: str,
                    include_metadata: bool = True) -> None:
        """Save analysis results to file.
        
        Args:
            results: Analysis results to save
            output_path: Output file path
            include_metadata: Include processing metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format based on file extension
        if output_path.suffix.lower() == '.json':
            self._save_json(results, output_path, include_metadata)
        elif output_path.suffix.lower() == '.csv':
            self._save_csv(results, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        logger.info(f"Results saved to: {output_path}")
    
    def _save_json(self, 
                   results: List[AnalysisResult], 
                   output_path: Path,
                   include_metadata: bool) -> None:
        """Save results as JSON."""
        data = {
            "metadata": {
                "total_texts": len(results),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": results[0].model_used if results else "unknown",
                "batch_size": self.batch_size,
                "max_workers": self.max_workers
            } if include_metadata else {},
            "results": [asdict(result) for result in results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, 
                  results: List[AnalysisResult], 
                  output_path: Path,
                  include_metadata: bool) -> None:
        """Save results as CSV."""
        if not results:
            return
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {
                "text": result.original_text,
                "sentiment": result.sentiment,
                "confidence": result.sentiment_confidence,
                "positive_score": result.sentiment_scores.get("positive", 0.0),
                "negative_score": result.sentiment_scores.get("negative", 0.0),
                "neutral_score": result.sentiment_scores.get("neutral", 0.0),
                "detected_language": result.detected_language,
                "language_confidence": result.language_confidence,
                "translation_needed": result.translation_needed,
                "translated_text": result.translated_text or "",
                "processing_time": result.processing_time
            }
            
            if include_metadata:
                row.update({
                    "model_used": result.model_used,
                    "cached_translation": result.cached_translation,
                    "preprocessed_text": result.preprocessed_text
                })
            
            csv_data.append(row)
        
        # Write CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress.
        
        Returns:
            Progress information
        """
        with self._lock:
            return {
                "processed_count": self._processed_count,
                "batch_size": self.batch_size,
                "max_workers": self.max_workers
            }
