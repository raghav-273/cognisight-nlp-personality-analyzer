# Cognisight 🧠

AI-Powered Conversation Personality Analyzer

## Features

- **Advanced Sentiment Analysis**: Uses state-of-the-art RoBERTa model for accurate sentiment detection
- **Big Five Personality Traits**: Analyzes text for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism
- **Interactive Visualizations**: Plotly charts for sentiment trends and personality scores
- **Word Cloud Generation**: Visual representation of most used words
- **Detailed Metrics**: Lexical diversity, readability scores, and more
- **Sample Texts**: Pre-loaded examples for quick testing

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8502`

## Deployment

This app can be deployed on Streamlit Community Cloud. Simply connect your GitHub repository and deploy.

## Technologies Used

- Streamlit for the web interface
- Hugging Face Transformers for NLP
- Plotly for interactive charts
- NLTK for text processing
- WordCloud for visualization

## Model

Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment analysis, providing 3-class sentiment (positive, neutral, negative).
