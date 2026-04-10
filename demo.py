#!/usr/bin/env python3
"""
Cognisight Demo Script
Demonstrates the enhanced personality analysis capabilities
"""

from analyzer import analyze_text
import json

def demo_analysis():
    """Run a comprehensive demo of the personality analysis system."""

    # Sample texts representing different personality types
    samples = {
        "Creative/Artistic": """
        Imagine a world where colors dance like fireflies in the twilight. What if our dreams could paint the sky with impossible hues?
        The symphony of thoughts creates melodies in my mind, each note a different shade of wonder and curiosity.
        """,

        "Professional/Business": """
        Dear Team, I am writing to inform you about the upcoming project deadline. Please ensure all tasks are completed by Friday.
        Let me know if you need any assistance with the implementation. We should schedule a follow-up meeting to discuss the results.
        """,

        "Friendly/Social": """
        Hi! How are you doing today? I'm great, thanks for asking. What about you? We should totally hang out sometime!
        I love meeting new people and hearing their stories. Life is so much better when we connect with others.
        """,

        "Analytical/Technical": """
        The algorithm implements a recursive backtracking approach with memoization to optimize the solution space.
        Time complexity is O(n^2) in worst case, but average case performance is significantly better with the heuristic optimization.
        """,

        "Emotional/Expressive": """
        I can't believe this happened! This is absolutely amazing and I'm so excited! Why does everything have to be so complicated?
        My heart is racing with joy and anticipation. This feeling is incredible!
        """
    }

    print("🧠 Cognisight Enhanced Personality Analysis Demo")
    print("=" * 60)

    for personality_type, text in samples.items():
        print(f"\n🎭 Analyzing: {personality_type}")
        print("-" * 40)

        # Perform analysis
        result = analyze_text(text.strip(), sentiment_model="combined", include_wordcloud=False)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue

        # Display key results
        personality = result["personality"]
        features = result["features"]
        sentiment = result["sentiment"]

        # Top 3 personality traits
        top_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)[:3]
        print("🏆 Top Personality Traits:")
        for trait, score in top_traits:
            print(f"  • {trait}: {score:.1f}")

        # Key metrics
        print("\n📊 Key Metrics:")
        print(f"  • Words: {features.get('word_count', 0)}")
        print(f"  • Sentences: {features.get('sentence_count', 0)}")
        print(f"  • Lexical Diversity: {features.get('lexical_diversity', 0):.2%}")
        print(f"  • Emotional Intensity: {features.get('emotional_intensity', 0):.1f}")

        # Communication style
        comm_style = features.get("communication_style", "unknown").title()
        print(f"💬 Communication Style: {comm_style}")

        # Sentiment summary
        avg_sent = sentiment.get("combined_avg", 0)
        sent_label = "Positive" if avg_sent > 0.1 else "Negative" if avg_sent < -0.1 else "Neutral"
        print(f"😊 Overall Sentiment: {sent_label} ({avg_sent:.2f})")
        print("-" * 40)

def compare_models():
    """Compare different sentiment analysis models."""
    print("\n🔍 Model Comparison Demo")
    print("=" * 40)

    test_text = "I'm really excited about this project! It's challenging but I'm confident we can succeed."

    models = ["basic", "advanced", "combined"]

    print(f"Text: '{test_text}'")
    print("\nModel Comparison:")

    for model in models:
        result = analyze_text(test_text, sentiment_model=model, include_wordcloud=False)
        if "error" not in result:
            sentiment = result["sentiment"]
            if model == "combined":
                score = sentiment.get("combined_avg", 0)
            elif model == "advanced":
                score = sentiment.get("advanced_avg", 0)
            else:
                score = sentiment.get("basic_avg", 0)

            print(f"  {model.title()}: {score:.3f}")
if __name__ == "__main__":
    try:
        demo_analysis()
        compare_models()
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()