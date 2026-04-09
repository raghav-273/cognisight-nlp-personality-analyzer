import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Download required resources
nltk.download('punkt')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load advanced sentiment model
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# ---------------- CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9.? ]', '', text)
    return text

def split_sentences(text):
    return sent_tokenize(text)

# ---------------- SENTIMENT ----------------
def get_sentiment_scores(sentences):
    scores = []
    for s in sentences:
        score = sia.polarity_scores(s)['compound']
        scores.append(score)
    return scores

def avg_sentiment(scores):
    if len(scores) == 0:
        return 0
    return sum(scores) / len(scores)

# ---------------- FEATURES ----------------
def personality_features(text):
    questions = text.count('?')
    statements = text.count('.')

    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return {
            "question_ratio": 0,
            "self_focus": 0,
            "social_focus": 0,
            "avg_word_length": 0,
            "word_count": 0,
            "lexical_diversity": 0,
            "readability_score": 0
        }

    pronouns = sum(1 for w in words if w in ['i','me','my','mine'])
    social = sum(1 for w in words if w in ['we','us','our','you','your'])

    avg_word_length = sum(len(w) for w in words) / total_words

    # Lexical diversity: unique words / total words
    unique_words = len(set(words))
    lexical_diversity = unique_words / total_words

    # Simple readability: average sentence length
    sentences = split_sentences(text)
    avg_sentence_length = total_words / len(sentences) if sentences else 0
    readability_score = 100 - avg_sentence_length  # Higher is better

    return {
        "question_ratio": questions / (statements + 1),
        "self_focus": pronouns,
        "social_focus": social,
        "avg_word_length": avg_word_length,
        "word_count": total_words,
        "lexical_diversity": lexical_diversity,
        "readability_score": readability_score
    }

# ---------------- PERSONALITY SCORES ----------------
def personality_scores(features, sentiment_avg):
    return {
        "Openness": round(features["lexical_diversity"] * 100, 2),
        "Conscientiousness": round(features["readability_score"], 2),
        "Extraversion": round(features["social_focus"] * 5 + abs(sentiment_avg) * 20, 2),
        "Agreeableness": round((1 - abs(sentiment_avg)) * 50 + features["question_ratio"] * 20, 2),
        "Neuroticism": round(abs(sentiment_avg) * 50, 2)
    }

# ---------------- SUMMARY ----------------
def generate_summary(features):
    summary = []

    if features["question_ratio"] > 0.5:
        summary.append("Highly inquisitive and curious communication style")
    else:
        summary.append("More declarative communication style")

    if features["self_focus"] > features["social_focus"]:
        summary.append("Tends to focus on personal experiences")
    else:
        summary.append("Emphasizes social interactions and others")

    if features["lexical_diversity"] > 0.7:
        summary.append("Uses a wide variety of vocabulary")
    else:
        summary.append("Uses repetitive language patterns")

    if features["readability_score"] > 50:
        summary.append("Communicates in clear, concise sentences")
    else:
        summary.append("Uses complex or lengthy sentence structures")

    return summary

# ---------------- EXPLANATION ----------------
def explain_output(features):
    explanations = []

    if features["question_ratio"] > 0.5:
        explanations.append("High question usage indicates curiosity and engagement")

    if features["lexical_diversity"] > 0.7:
        explanations.append("Diverse vocabulary suggests broad knowledge or creativity")

    if features["social_focus"] > features["self_focus"]:
        explanations.append("Focus on others indicates social orientation")

    if features["readability_score"] > 50:
        explanations.append("Clear sentences suggest effective communication skills")

    return explanations

# ---------------- WORD CLOUD ----------------
def generate_wordcloud(text):
    if not text.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"