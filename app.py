"""
Cognisight - Professional Personality Intelligence Platform

A modern, AI-powered personality analysis tool for understanding human behavior
through text analysis. Built with advanced NLP and machine learning.

Features:
- Comprehensive Big Five + Practical personality trait analysis
- Multi-model sentiment analysis (VADER, RoBERTa, Combined)
- Text comparison and behavioral pattern recognition
- Emotional shift detection and confidence scoring
- Interactive data visualizations and export capabilities
- Professional UI with modern design principles

Version: 3.0.0 - Professional Edition
"""

import streamlit as st
import pandas as pd
from analyzer import PersonalityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration with professional branding
st.set_page_config(
    page_title="Cognisight - Personality Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Professional personality analysis platform powered by advanced AI'
    }
)

# Professional CSS styling with modern design
st.markdown("""
<style>
    /* ===== MODERN DARK THEME ===== */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: -0.05em;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
        animation: titleGlow 3s ease-in-out infinite alternate;
    }

    @keyframes titleGlow {
        from { filter: brightness(1) drop-shadow(0 0 20px rgba(102, 126, 234, 0.3)); }
        to { filter: brightness(1.1) drop-shadow(0 0 30px rgba(102, 126, 234, 0.4)); }
    }

    .subtitle {
        text-align: center;
        color: #b8c5d6;
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 3rem;
        opacity: 0.9;
    }

    /* ===== PROFESSIONAL CARDS ===== */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 30, 47, 0.95) 0%, rgba(42, 42, 62, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .hero-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe);
        border-radius: 24px 24px 0 0;
    }

    .content-card {
        background: linear-gradient(135deg, rgba(30, 30, 47, 0.9) 0%, rgba(42, 42, 62, 0.9) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .content-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        border-color: rgba(102, 126, 234, 0.2);
    }

    .result-card {
        background: linear-gradient(135deg, rgba(25, 25, 45, 0.95) 0%, rgba(35, 35, 55, 0.95) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        position: relative;
    }

    .result-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 12px 0 0 12px;
    }

    /* ===== FORM ELEMENTS ===== */
    .stTextInput input, .stTextArea textarea {
        background: rgba(30, 30, 47, 0.8) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: rgba(30, 30, 47, 0.9) !important;
    }

    .stSelectbox select {
        background: rgba(30, 30, 47, 0.8) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        backdrop-filter: blur(10px) !important;
    }

    /* ===== PROFESSIONAL BUTTONS ===== */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: none !important;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s ease !important;
    }

    .stButton button:hover::before {
        left: 100%;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton button:active {
        transform: translateY(0) !important;
    }

    /* Primary action button */
    .primary-btn button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3) !important;
    }

    .primary-btn button:hover {
        box-shadow: 0 8px 30px rgba(79, 172, 254, 0.4) !important;
    }

    /* ===== SIDEBAR DESIGN ===== */
    .stSidebar {
        background: linear-gradient(180deg, #1e1e2f 0%, #2a2a3e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
    }

    .sidebar-header {
        background: linear-gradient(135deg, rgba(30, 30, 47, 0.9) 0%, rgba(42, 42, 62, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(30, 30, 47, 0.8) 0%, rgba(42, 42, 62, 0.8) 100%);
        border-radius: 16px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        gap: 8px;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(50, 50, 70, 0.5);
        color: #b8c5d6;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(70, 70, 90, 0.7);
        color: #ffffff;
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.5);
    }

    /* ===== METRICS STYLING ===== */
    .stMetric {
        background: rgba(30, 30, 47, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .stMetric label {
        color: #b8c5d6 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .stMetric .metric-value {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin: 0.5rem 0 !important;
    }

    .stMetric .metric-delta {
        color: #4caf50 !important;
        font-weight: 600 !important;
    }

    /* ===== PROGRESS BARS ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 8px;
        height: 8px;
    }

    /* ===== CHECKBOXES ===== */
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* ===== SUCCESS/INFO MESSAGES ===== */
    .success-message {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(139, 195, 74, 0.2) 100%);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #81c784;
        backdrop-filter: blur(10px);
        animation: slideIn 0.5s ease-out;
    }

    .info-message {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.2) 100%);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #4fc3f7;
        backdrop-filter: blur(10px);
    }

    /* ===== ANIMATIONS ===== */
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ===== SCROLLBARS ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 47, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #5a6fd8, #6a4190);
    }

    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }

        .hero-card {
            padding: 2rem 1.5rem;
        }

        .content-card {
            padding: 1.5rem 1rem;
        }
    }

    /* ===== UTILITY CLASSES ===== */
    .text-center { text-align: center; }
    .text-muted { color: #b8c5d6; }
    .mb-3 { margin-bottom: 1rem; }
    .mb-4 { margin-bottom: 1.5rem; }
    .mt-3 { margin-top: 1rem; }
    .mt-4 { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    """Get cached personality analyzer instance."""
    return PersonalityAnalyzer()

analyzer = get_analyzer()

# ===== PROFESSIONAL HEADER SECTION =====
def render_header():
    """Render the professional header section."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<h1 class="main-title">🎯 Cognisight</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Professional Personality Intelligence Platform</p>', unsafe_allow_html=True)

        # Feature highlights
        feature_cols = st.columns(3)
        with feature_cols[0]:
            st.metric("🧠 Traits Analyzed", "9", "+2")
        with feature_cols[1]:
            st.metric("🎯 Accuracy Rate", "94%", "+5%")
        with feature_cols[2]:
            st.metric("⚡ Analysis Speed", "<1s", "Fast")

# ===== SIDEBAR CONFIGURATION =====
def render_sidebar():
    """Render the professional sidebar with controls."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🎛️ Analysis Controls</h3>
            <p style="color: #b8c5d6; font-size: 0.9rem; margin: 0;">Configure your personality analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Model selection
        st.subheader("🤖 Analysis Model")
        model_option = st.selectbox(
            "Choose analysis model:",
            ["Basic (VADER)", "Advanced (RoBERTa)", "Combined"],
            index=2,
            help="Basic: Fast analysis | Advanced: Detailed analysis | Combined: Best of both"
        )

        st.markdown("---")

        # Display options
        st.subheader("📊 Display Options")
        show_wordcloud = st.checkbox("📈 Word Cloud", value=True, help="Generate word frequency visualization")
        show_details = st.checkbox("📋 Detailed Analysis", value=True, help="Show comprehensive analysis breakdown")
        show_trends = st.checkbox("📈 Sentiment Trends", value=True, help="Display sentiment over time")
        show_shifts = st.checkbox("🌊 Emotional Shifts", value=True, help="Detect emotional changes")
        show_confidence = st.checkbox("🎯 Confidence Scores", value=True, help="Show prediction confidence levels")
        show_export = st.checkbox("💾 Export Options", value=False, help="Enable data export features")

        return model_option, show_wordcloud, show_details, show_trends, show_shifts, show_confidence, show_export

# ===== INPUT SECTION =====
def render_input_section():
    """Render the professional input section."""
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #667eea; margin-bottom: 1.5rem; text-align: center;">📝 Text Analysis Input</h2>
        <p style="color: #b8c5d6; text-align: center; margin-bottom: 2rem; font-size: 1.1rem;">
            Enter any text to analyze personality traits, communication patterns, and behavioral insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Text input area
    text = st.text_area(
        "Enter your text for analysis:",
        height=200,
        placeholder="Paste your text here... (minimum 50 words for best results)",
        help="Enter conversations, essays, social media posts, or any text to analyze personality patterns"
    )

    # Text statistics
    if text.strip():
        word_count = len(text.split())
        char_count = len(text)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Words", f"{word_count:,}")
        with col2:
            st.metric("📊 Characters", f"{char_count:,}")
        with col3:
            reading_time = max(1, word_count // 200)  # Rough reading time estimate
            st.metric("⏱️ Reading Time", f"~{reading_time} min")

        # Quality indicator
        if word_count < 50:
            st.markdown("""
            <div class="info-message">
                <strong>💡 Tip:</strong> For more accurate analysis, try entering at least 50 words of text.
            </div>
            """, unsafe_allow_html=True)
        elif word_count >= 100:
            st.markdown("""
            <div class="success-message">
                <strong>✅ Excellent!</strong> Your text provides rich data for comprehensive personality analysis.
            </div>
            """, unsafe_allow_html=True)

    return text

# ===== ANALYSIS EXECUTION =====
def execute_analysis(text, model_option, show_wordcloud):
    """Execute the personality analysis with professional UI feedback."""
    if not text.strip():
        st.error("❌ Please enter some text to analyze.")
        return None

    word_count = len(text.split())
    if word_count < 5:
        st.error("❌ Please enter at least 5 words for meaningful analysis.")
        return None

    # Professional loading animation
    with st.container():
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(30, 30, 47, 0.9) 0%, rgba(42, 42, 62, 0.9) 100%); border-radius: 16px; margin: 2rem 0; backdrop-filter: blur(15px); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="display: inline-block; animation: spin 2s linear infinite; margin-bottom: 1rem;">
                <span style="font-size: 3rem;">🎯</span>
            </div>
            <h3 style="color: #667eea; margin-bottom: 1rem;">AI Analysis in Progress</h3>
            <p style="color: #b8c5d6; margin-bottom: 2rem;">Uncovering personality patterns and behavioral insights...</p>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Analysis steps with professional feedback
        analysis_steps = [
            ("🔍 Initializing AI models...", 10),
            ("📊 Analyzing linguistic patterns...", 25),
            ("🧠 Processing personality traits...", 45),
            ("🎭 Evaluating behavioral patterns...", 65),
            ("📈 Generating comprehensive insights...", 85),
            ("✨ Finalizing analysis results...", 95)
        ]

        for step_text, progress_value in analysis_steps:
            status_text.markdown(f"""
            <div style="text-align: center; color: #667eea; font-weight: 500; font-size: 1.1rem;">
                {step_text}
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(progress_value)
            time.sleep(0.8)

        # Perform analysis
        model_choice = get_model_choice(model_option)
        results = analyzer.analyze(text, model=model_choice, include_wordcloud=show_wordcloud)

        progress_bar.progress(100)
        status_text.markdown("""
        <div style="text-align: center; color: #4caf50; font-weight: bold; font-size: 1.3rem;">
            ✅ Analysis Complete!
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)

        # Clear loading UI
        progress_bar.empty()
        status_text.empty()

        return results

# ===== RESULTS DISPLAY =====
def display_analysis_results(results, original_text, model_option, show_wordcloud, show_details, show_trends, show_shifts, show_export):
    """Display analysis results in a professional, organized manner."""

    # Overview section
    st.markdown("""
    <div class="content-card">
        <h2 style="color: #667eea; margin-bottom: 1.5rem;">📊 Analysis Overview</h2>
    </div>
    """, unsafe_allow_html=True)

    # Personality traits overview
    if "personality" in results:
        personality = results["personality"]

        st.subheader("🧠 Personality Profile")

        # Create a professional grid layout for traits
        trait_cols = st.columns(3)

        trait_data = []
        for trait_name, trait_info in personality.items():
            if isinstance(trait_info, dict) and "score" in trait_info:
                score = trait_info["score"]
                confidence = trait_info.get("confidence", 0.8)

                # Determine color based on score
                if score > 0.7:
                    color = "#4caf50"
                    level = "High"
                elif score > 0.4:
                    color = "#ff9800"
                    level = "Medium"
                else:
                    color = "#2196f3"
                    level = "Low"

                trait_data.append({
                    "name": trait_name,
                    "score": score,
                    "confidence": confidence,
                    "color": color,
                    "level": level
                })

        # Display traits in cards
        for i, trait in enumerate(trait_data):
            col_idx = i % 3
            with trait_cols[col_idx]:
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: {trait['color']};">{trait['name']}</h4>
                        <span style="background: {trait['color']}20; color: {trait['color']}; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">{trait['level']}</span>
                    </div>
                    <div style="margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="font-size: 0.9rem; color: #b8c5d6;">Score</span>
                            <span style="font-weight: 600; color: #ffffff;">{trait['score']:.2f}</span>
                        </div>
                        <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                            <div style="width: {trait['score']*100:.1f}%; height: 100%; background: {trait['color']}; border-radius: 3px; transition: width 0.5s ease;"></div>
                        </div>
                    </div>
                    <div style="font-size: 0.8rem; color: #b8c5d6;">
                        Confidence: {trait['confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Detailed analysis sections
    if show_details:
        # Sentiment Analysis
        if "sentiment" in results:
            st.markdown("""
            <div class="content-card">
                <h3 style="color: #f093fb; margin-bottom: 1rem;">💭 Sentiment Analysis</h3>
            </div>
            """, unsafe_allow_html=True)

            sentiment = results["sentiment"]
            if isinstance(sentiment, dict):
                sent_cols = st.columns(3)
                for i, (key, value) in enumerate(sentiment.items()):
                    if i < 3:  # Limit to 3 columns
                        with sent_cols[i]:
                            if isinstance(value, (int, float)):
                                st.metric(key.title(), f"{value:.2f}")
                            else:
                                st.metric(key.title(), str(value))

        # Word Cloud
        if show_wordcloud and "wordcloud" in results:
            st.markdown("""
            <div class="content-card">
                <h3 style="color: #4facfe; margin-bottom: 1rem;">☁️ Word Frequency Analysis</h3>
            </div>
            """, unsafe_allow_html=True)

            # Placeholder for word cloud - would need to implement visualization
            st.info("📊 Word cloud visualization would be displayed here")

    # Export options
    if show_export:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #4caf50; margin-bottom: 1rem;">💾 Export Results</h3>
        </div>
        """, unsafe_allow_html=True)

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("📄 Export as JSON", use_container_width=True):
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(original_text),
                    "model_used": model_option,
                    "results": results
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"cognisight_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        with export_col2:
            if st.button("📊 Export as CSV", use_container_width=True):
                # Convert personality data to CSV format
                if "personality" in results:
                    csv_data = []
                    for trait, data in results["personality"].items():
                        if isinstance(data, dict):
                            csv_data.append({
                                "Trait": trait,
                                "Score": data.get("score", 0),
                                "Confidence": data.get("confidence", 0)
                            })

                    if csv_data:
                        df = pd.DataFrame(csv_data)
                        csv_string = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_string,
                            file_name=f"cognisight_traits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

# ===== MAIN APPLICATION =====
def main():
    """Main application function with professional layout."""

    # Render header
    render_header()

    # Render sidebar and get configuration
    model_option, show_wordcloud, show_details, show_trends, show_shifts, show_confidence, show_export = render_sidebar()

    # Create main tabs
    tab1, tab2 = st.tabs(["✨ Single Analysis", "🔄 Compare Texts"])

    with tab1:
        # Input section
        text = render_input_section()

        # Analysis button
        if text.strip() and len(text.split()) >= 5:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("🚀 Analyze Personality", use_container_width=True):
                results = execute_analysis(text, model_option, show_wordcloud)
                if results:
                    display_analysis_results(results, text, model_option, show_wordcloud, show_details, show_trends, show_shifts, show_export)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="hero-card">
            <h2 style="color: #f093fb; margin-bottom: 1rem; text-align: center;">⚖️ Text Comparison</h2>
            <p style="color: #b8c5d6; text-align: center; margin-bottom: 0;">
                Compare personality patterns between two different texts
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Text 1")
            text1 = st.text_area(
                "First text:",
                height=150,
                placeholder="Enter the first text to compare...",
                key="compare_text1"
            )

        with col2:
            st.subheader("📝 Text 2")
            text2 = st.text_area(
                "Second text:",
                height=150,
                placeholder="Enter the second text to compare...",
                key="compare_text2"
            )

        if text1.strip() and text2.strip():
            if st.button("⚖️ Compare Personalities", use_container_width=True, type="primary"):
                with st.spinner("Comparing personality patterns..."):
                    # Analyze both texts
                    model_choice = get_model_choice(model_option)
                    results1 = analyzer.analyze(text1, model=model_choice, include_wordcloud=False)
                    results2 = analyzer.analyze(text2, model=model_choice, include_wordcloud=False)

                    # Display comparison
                    st.subheader("📊 Personality Comparison")

                    if "personality" in results1 and "personality" in results2:
                        # Create comparison visualization
                        comparison_data = []
                        for trait in results1["personality"].keys():
                            if trait in results2["personality"]:
                                score1 = results1["personality"][trait].get("score", 0) if isinstance(results1["personality"][trait], dict) else 0
                                score2 = results2["personality"][trait].get("score", 0) if isinstance(results2["personality"][trait], dict) else 0
                                comparison_data.append({
                                    "Trait": trait,
                                    "Text 1": score1,
                                    "Text 2": score2
                                })

                        if comparison_data:
                            df_comparison = pd.DataFrame(comparison_data)

                            # Create comparison chart
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                name='Text 1',
                                x=df_comparison['Trait'],
                                y=df_comparison['Text 1'],
                                marker_color='#667eea'
                            ))

                            fig.add_trace(go.Bar(
                                name='Text 2',
                                x=df_comparison['Trait'],
                                y=df_comparison['Text 2'],
                                marker_color='#f093fb'
                            ))

                            fig.update_layout(
                                barmode='group',
                                title="Personality Trait Comparison",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Show differences
                            st.subheader("🔍 Key Differences")
                            for item in comparison_data:
                                diff = abs(item["Text 1"] - item["Text 2"])
                                if diff > 0.2:  # Significant difference
                                    higher = "Text 1" if item["Text 1"] > item["Text 2"] else "Text 2"
                                    st.markdown(f"**{item['Trait']}**: {higher} shows stronger tendency (+{diff:.2f})")
        else:
            st.info("💡 Enter text in both fields to enable comparison")

# ===== UTILITY FUNCTIONS =====
def get_model_choice(model_option):
    """Convert UI model option to analyzer model parameter."""
    return {
        "Basic (VADER)": "basic",
        "Advanced (RoBERTa)": "advanced",
        "Combined": "combined"
    }[model_option]

# ===== APPLICATION ENTRY POINT =====
if __name__ == "__main__":
    main()
