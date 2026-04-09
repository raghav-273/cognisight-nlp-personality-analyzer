import streamlit as st
import pandas as pd
from analyzer import *
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Cognisight", layout="wide", page_icon="🧠")

st.title("🧠 Cognisight")
st.subheader("AI-Powered Conversation Personality Analyzer")

# Sidebar for settings
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox("Sentiment Model", ["Basic", "Advanced"], index=1)
show_wordcloud = st.sidebar.checkbox("Show Word Cloud", value=True)
show_details = st.sidebar.checkbox("Show Detailed Analysis", value=True)

# Sample texts
sample_texts = {
    "Friendly Chat": "Hi! How are you doing today? I'm great, thanks for asking. What about you?",
    "Professional Email": "Dear Team, I am writing to inform you about the upcoming project deadline. Please ensure all tasks are completed by Friday.",
    "Argumentative": "I can't believe you said that! That's completely wrong and you know it. Why would you even think that?",
    "Reflective": "I've been thinking a lot about my life lately. What does it all mean? I wonder about the future."
}

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Enter conversation text:", height=200, placeholder="Paste your conversation here...")
with col2:
    st.subheader("Sample Texts")
    for name, sample in sample_texts.items():
        if st.button(name, key=name):
            text = sample
            st.rerun()

if st.button("Analyze", type="primary") and text.strip():
    with st.spinner("Analyzing..."):
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)

        scores = get_sentiment_scores(sentences)
        avg_sent = avg_sentiment(scores)

        features = personality_features(cleaned)
        summary = generate_summary(features)
        explanations = explain_output(features)
        trait_scores = personality_scores(features, avg_sent)

        wordcloud_img = generate_wordcloud(cleaned) if show_wordcloud else None

    # ---------------- LAYOUT ----------------
    st.success("Analysis Complete!")

    # Sentiment Analysis
    st.header("📊 Sentiment Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Trend")
        if scores:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=scores, mode='lines+markers', name='Sentiment'))
            fig.update_layout(xaxis_title="Sentence", yaxis_title="Sentiment Score", yaxis=dict(range=[-1.1, 1.1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentences detected.")

    with col2:
        st.subheader("Overall Sentiment")
        if avg_sent > 0.1:
            st.success(f"Positive Tone ({avg_sent:.2f})")
        elif avg_sent < -0.1:
            st.error(f"Negative Tone ({avg_sent:.2f})")
        else:
            st.info(f"Neutral Tone ({avg_sent:.2f})")

    # Personality Traits
    st.header("🧠 Personality Traits (Big Five)")
    df = pd.DataFrame(list(trait_scores.items()), columns=["Trait", "Score"])
    fig = px.bar(df, x="Trait", y="Score", color="Trait", title="Personality Scores")
    fig.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

    # Word Cloud
    if wordcloud_img:
        st.header("☁️ Word Cloud")
        st.image(wordcloud_img, use_column_width=True)

    # Summary and Explanations
    st.header("📋 Analysis Summary")
    for s in summary:
        st.write("•", s)

    if show_details:
        st.header("🔍 Detailed Explanations")
        for e in explanations:
            st.write("•", e)

        st.header("📈 Key Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Total Words", features["word_count"])
            st.metric("Questions", int(features["question_ratio"] * (features["word_count"] / 10)))
        with metrics_col2:
            st.metric("Lexical Diversity", f"{features['lexical_diversity']:.2f}")
            st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
        with metrics_col3:
            st.metric("Readability Score", f"{features['readability_score']:.1f}")
            st.metric("Sentences", len(sentences))

elif not text.strip():
    st.warning("Please enter some text to analyze.")
else:
    st.info("Enter conversation text and click Analyze to get insights.")

st.markdown("---")
st.caption("Built with Streamlit, Transformers, and NLP analysis. Model: Cardiff NLP Twitter RoBERTa Sentiment.")