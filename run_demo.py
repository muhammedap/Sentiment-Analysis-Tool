#!/usr/bin/env python3
"""
Demo script for the Multilingual Sentiment Analysis Tool.

This script demonstrates the core functionality of the sentiment analysis system
including single text analysis, batch processing, and multilingual support.
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.core.multilingual_analyzer import MultilingualAnalyzer
from app.core.language_detector import LanguageDetector
from app.core.translator import TranslationService
from app.utils.batch_processor import BatchProcessor


def print_separator(title=""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


def demo_single_analysis():
    """Demonstrate single text analysis."""
    print_separator("Single Text Analysis Demo")
    
    # Initialize analyzer
    print("Initializing multilingual analyzer...")
    analyzer = MultilingualAnalyzer()
    
    # Test texts in different languages
    test_texts = [
        ("I love this product! It's amazing.", "English"),
        ("Me encanta este producto! Es increíble.", "Spanish"),
        ("J'adore ce produit! C'est incroyable.", "French"),
        ("Ich liebe dieses Produkt! Es ist erstaunlich.", "German"),
        ("我喜欢这个产品！太棒了。", "Chinese"),
        ("This product is terrible. I hate it.", "English (Negative)"),
        ("The weather is okay today.", "English (Neutral)")
    ]
    
    for text, description in test_texts:
        print(f"\nAnalyzing {description}:")
        print(f"Text: {text}")
        
        start_time = time.time()
        result = analyzer.analyze_text(text)
        processing_time = time.time() - start_time
        
        print(f"Language: {result.detected_language} (confidence: {result.language_confidence:.3f})")
        print(f"Sentiment: {result.sentiment} (confidence: {result.sentiment_confidence:.3f})")
        print(f"Scores: Pos={result.sentiment_scores.get('positive', 0):.3f}, "
              f"Neg={result.sentiment_scores.get('negative', 0):.3f}, "
              f"Neu={result.sentiment_scores.get('neutral', 0):.3f}")
        print(f"Translation needed: {result.translation_needed}")
        if result.translated_text:
            print(f"Translated: {result.translated_text}")
        print(f"Processing time: {processing_time:.3f}s")


def demo_language_detection():
    """Demonstrate language detection."""
    print_separator("Language Detection Demo")
    
    detector = LanguageDetector()
    
    test_texts = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hallo, wie geht es dir heute?",
        "你好，你今天怎么样？",
        "Ciao, come stai oggi?",
        "Olá, como você está hoje?",
        "Привет, как дела сегодня?",
        "こんにちは、今日はいかがですか？",
        "مرحبا، كيف حالك اليوم؟"
    ]
    
    print("Detecting languages for various texts:")
    for text in test_texts:
        result = detector.detect_language(text)
        language_name = detector.get_language_name(result.language)
        supported = "✓" if result.is_supported else "✗"
        
        print(f"Text: {text[:30]}...")
        print(f"  Language: {result.language} ({language_name}) - {supported} Supported")
        print(f"  Confidence: {result.confidence:.3f}")
        if result.alternatives:
            alt_str = ", ".join([f"{lang}({conf:.2f})" for lang, conf in result.alternatives[:3]])
            print(f"  Alternatives: {alt_str}")
        print()


def demo_batch_processing():
    """Demonstrate batch processing with CSV file."""
    print_separator("Batch Processing Demo")
    
    # Check if sample data exists
    csv_file = Path("data/sample_reviews.csv")
    if not csv_file.exists():
        print(f"Sample data file not found: {csv_file}")
        print("Creating sample data...")
        
        # Create sample data
        sample_data = {
            "text": [
                "I love this product! Amazing quality.",
                "Terrible service. Very disappointed.",
                "The product is okay, nothing special.",
                "¡Excelente producto! Lo recomiendo mucho.",
                "Service client horrible. Je ne recommande pas."
            ],
            "language": ["en", "en", "en", "es", "fr"],
            "expected_sentiment": ["positive", "negative", "neutral", "positive", "negative"]
        }
        
        df = pd.DataFrame(sample_data)
        csv_file.parent.mkdir(exist_ok=True)
        df.to_csv(csv_file, index=False)
        print(f"Sample data created at: {csv_file}")
    
    # Initialize components
    analyzer = MultilingualAnalyzer()
    
    def progress_callback(processed, total):
        percentage = (processed / total) * 100
        print(f"Progress: {processed}/{total} ({percentage:.1f}%)")
    
    processor = BatchProcessor(
        analyzer, 
        batch_size=8, 
        max_workers=2,
        progress_callback=progress_callback
    )
    
    print(f"Processing CSV file: {csv_file}")
    start_time = time.time()
    
    # Process the CSV file
    results = processor.process_csv_file(
        str(csv_file),
        text_column="text",
        output_path="results/demo_results.csv",
        include_metadata=True
    )
    
    total_time = time.time() - start_time
    
    print(f"\nBatch processing completed!")
    print(f"Processed {len(results)} texts in {total_time:.2f}s")
    print(f"Average time per text: {total_time/len(results):.3f}s")
    
    # Show summary statistics
    sentiments = [r.sentiment for r in results]
    languages = [r.detected_language for r in results]
    translations_needed = sum(1 for r in results if r.translation_needed)
    
    print(f"\nSummary:")
    print(f"  Positive: {sentiments.count('positive')}")
    print(f"  Negative: {sentiments.count('negative')}")
    print(f"  Neutral: {sentiments.count('neutral')}")
    print(f"  Languages detected: {set(languages)}")
    print(f"  Translations needed: {translations_needed}")
    
    # Show first few results
    print(f"\nFirst 3 results:")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. Text: {result.original_text[:50]}...")
        print(f"     Sentiment: {result.sentiment} ({result.sentiment_confidence:.3f})")
        print(f"     Language: {result.detected_language}")
        if result.translation_needed:
            print(f"     Translated: {result.translated_text[:50]}...")


def demo_translation():
    """Demonstrate translation functionality."""
    print_separator("Translation Demo")
    
    translator = TranslationService()
    
    test_cases = [
        ("Hola, ¿cómo estás?", "es", "en"),
        ("Bonjour, comment allez-vous?", "fr", "en"),
        ("Wie geht es dir?", "de", "en"),
        ("Come stai?", "it", "en"),
        ("Como você está?", "pt", "en")
    ]
    
    print("Testing translation service:")
    for text, source_lang, target_lang in test_cases:
        result = translator.translate_text(text, source_lang, target_lang)
        
        print(f"\nOriginal ({source_lang}): {result.original_text}")
        print(f"Translated ({target_lang}): {result.translated_text}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Cached: {result.cached}")


def demo_system_info():
    """Display system information."""
    print_separator("System Information")
    
    try:
        analyzer = MultilingualAnalyzer()
        system_info = analyzer.get_system_info()
        
        print("System Configuration:")
        for key, value in system_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error getting system info: {e}")


def main():
    """Run all demos."""
    print("🌍 Multilingual Sentiment Analysis Tool - Demo")
    print("=" * 60)
    print("Repository: https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool")
    print("=" * 60)
    
    try:
        # Run demos
        demo_system_info()
        demo_language_detection()
        demo_single_analysis()
        demo_translation()
        demo_batch_processing()
        
        print_separator("Demo Completed Successfully!")
        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. 🌐 Try the live demo: https://multilingual-sentiment-analysis.streamlit.app/")
        print("2. Start the API server: uvicorn app.api.main:app --reload")
        print("3. Start the web interface: streamlit run app/frontend/streamlit_app.py")
        print("4. Check the results in the 'results/' directory")
        print("5. Explore the API documentation at http://localhost:8000/docs")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or model download issues.")
        print("Please check the installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
