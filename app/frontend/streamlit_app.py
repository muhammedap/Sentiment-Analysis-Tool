"""Streamlit web interface for multilingual sentiment analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import json
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import core components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.multilingual_analyzer import MultilingualAnalyzer, AnalysisResult


@st.cache_resource
def load_analyzer():
    """Load and cache the sentiment analyzer."""
    with st.spinner("Loading sentiment analysis model..."):
        analyzer = MultilingualAnalyzer()
    return analyzer


def create_sentiment_chart(results: List[AnalysisResult]) -> go.Figure:
    """Create sentiment distribution chart."""
    sentiments = [result.sentiment for result in results]
    sentiment_counts = pd.Series(sentiments).value_counts()

    colors = {
        'positive': '#2E8B57',
        'negative': '#DC143C',
        'neutral': '#4682B4'
    }

    # Create color map for all sentiments, using default colors for unknown ones
    color_map = {}
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, sentiment in enumerate(sentiment_counts.index):
        if sentiment in colors:
            color_map[sentiment] = colors[sentiment]
        else:
            color_map[sentiment] = default_colors[i % len(default_colors)]

    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=color_map
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig


def create_confidence_chart(results: List[AnalysisResult]) -> go.Figure:
    """Create confidence distribution chart."""
    confidences = [result.sentiment_confidence for result in results]
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Count'}
    )
    
    fig.update_layout(height=400)
    return fig


def create_language_chart(results: List[AnalysisResult]) -> go.Figure:
    """Create language distribution chart."""
    languages = [result.detected_language for result in results]
    lang_counts = pd.Series(languages).value_counts()
    
    fig = px.bar(
        x=lang_counts.index,
        y=lang_counts.values,
        title="Language Distribution",
        labels={'x': 'Language', 'y': 'Count'}
    )
    
    fig.update_layout(height=400)
    return fig


def format_analysis_result(result: AnalysisResult) -> Dict[str, Any]:
    """Format analysis result for display."""
    return {
        "Text": result.original_text[:100] + "..." if len(result.original_text) > 100 else result.original_text,
        "Sentiment": result.sentiment.title(),
        "Confidence": f"{result.sentiment_confidence:.3f}",
        "Language": result.detected_language.upper(),
        "Translation Needed": "Yes" if result.translation_needed else "No",
        "Processing Time": f"{result.processing_time:.3f}s"
    }


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🌍 Multilingual Sentiment Analysis Tool")
    st.markdown("""
    Analyze sentiment in text across multiple languages using state-of-the-art transformer models.
    Supports automatic language detection and translation fallback for unsupported languages.
    """)
    
    # Load analyzer
    try:
        analyzer = load_analyzer()
        st.success("✅ Sentiment analyzer loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load sentiment analyzer: {e}")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    enable_translation = st.sidebar.checkbox(
        "Enable Translation", 
        value=True,
        help="Automatically translate unsupported languages"
    )
    
    enable_preprocessing = st.sidebar.checkbox(
        "Enable Preprocessing",
        value=True, 
        help="Clean and preprocess text before analysis"
    )
    
    # Configure analyzer
    analyzer.enable_translation = enable_translation
    analyzer.enable_preprocessing = enable_preprocessing
    
    # Display system info
    with st.sidebar.expander("📊 System Information"):
        system_info = analyzer.get_system_info()
        st.json(system_info)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📄 Batch Analysis", "📈 Analytics"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary")
        
        if analyze_button and text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    result = analyzer.analyze_text(text_input)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_color_map = {
                            'positive': 'green',
                            'negative': 'red',
                            'neutral': 'blue'
                        }
                        sentiment_color = sentiment_color_map.get(result.sentiment, 'gray')

                        st.metric(
                            "Sentiment",
                            result.sentiment.title(),
                            delta=None
                        )
                        st.markdown(f"<span style='color: {sentiment_color}'>●</span>", unsafe_allow_html=True)
                    
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{result.sentiment_confidence:.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Language",
                            result.detected_language.upper()
                        )
                    
                    # Detailed scores
                    st.subheader("📊 Detailed Scores")
                    scores_df = pd.DataFrame([
                        {"Sentiment": "Positive", "Score": result.sentiment_scores.get("positive", 0)},
                        {"Sentiment": "Negative", "Score": result.sentiment_scores.get("negative", 0)},
                        {"Sentiment": "Neutral", "Score": result.sentiment_scores.get("neutral", 0)}
                    ])
                    
                    fig = px.bar(
                        scores_df,
                        x="Sentiment",
                        y="Score",
                        color="Sentiment",
                        color_discrete_map={
                            'Positive': '#2E8B57',
                            'Negative': '#DC143C',
                            'Neutral': '#4682B4'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional information
                    with st.expander("🔍 Additional Information"):
                        info_data = {
                            "Original Text": result.original_text,
                            "Detected Language": f"{result.detected_language} ({result.language_confidence:.3f} confidence)",
                            "Translation Needed": "Yes" if result.translation_needed else "No",
                            "Translated Text": result.translated_text or "N/A",
                            "Processing Time": f"{result.processing_time:.3f} seconds",
                            "Model Used": result.model_used
                        }
                        
                        for key, value in info_data.items():
                            st.write(f"**{key}:** {value}")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with texts to analyze:",
            type=['csv'],
            help="CSV should have a 'text' column"
        )
        
        # Manual text input
        st.subheader("Or enter multiple texts manually:")
        manual_texts = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="Text 1\nText 2\nText 3..."
        )
        
        if st.button("🔍 Analyze Batch", type="primary"):
            texts_to_analyze = []
            
            # Process uploaded file
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts_to_analyze.extend(df['text'].dropna().tolist())
                    else:
                        st.error("❌ CSV file must have a 'text' column")
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {e}")
            
            # Process manual texts
            if manual_texts.strip():
                manual_list = [text.strip() for text in manual_texts.split('\n') if text.strip()]
                texts_to_analyze.extend(manual_list)
            
            if texts_to_analyze:
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    try:
                        results = analyzer.analyze_batch(texts_to_analyze)
                        
                        # Display summary
                        st.success(f"✅ Analyzed {len(results)} texts successfully!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        sentiments = [r.sentiment for r in results]
                        
                        with col1:
                            st.metric("Total Texts", len(results))
                        
                        with col2:
                            st.metric("Positive", sentiments.count('positive'))
                        
                        with col3:
                            st.metric("Negative", sentiments.count('negative'))
                        
                        with col4:
                            st.metric("Neutral", sentiments.count('neutral'))
                        
                        # Results table
                        st.subheader("📋 Results")
                        results_df = pd.DataFrame([format_analysis_result(r) for r in results])
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Store results in session state for analytics
                        st.session_state['batch_results'] = results
                        
                    except Exception as e:
                        st.error(f"❌ Batch analysis failed: {e}")
            else:
                st.warning("⚠️ Please upload a CSV file or enter texts manually.")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if 'batch_results' in st.session_state and st.session_state['batch_results']:
            results = st.session_state['batch_results']
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_fig = create_sentiment_chart(results)
                st.plotly_chart(sentiment_fig, use_container_width=True)
            
            with col2:
                confidence_fig = create_confidence_chart(results)
                st.plotly_chart(confidence_fig, use_container_width=True)
            
            # Language distribution
            language_fig = create_language_chart(results)
            st.plotly_chart(language_fig, use_container_width=True)
            
            # Statistics
            st.subheader("📊 Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_confidence = sum(r.sentiment_confidence for r in results) / len(results)
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            with col2:
                translations_needed = sum(1 for r in results if r.translation_needed)
                st.metric("Translations Needed", f"{translations_needed}/{len(results)}")
            
            with col3:
                avg_processing_time = sum(r.processing_time for r in results) / len(results)
                st.metric("Avg Processing Time", f"{avg_processing_time:.3f}s")
            
        else:
            st.info("📈 Run a batch analysis to see analytics here.")


if __name__ == "__main__":
    main()
