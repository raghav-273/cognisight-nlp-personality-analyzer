"""
Cognisight Streamlit application - Redesigned for clarity and impact.

A calm, structured journaling intelligence tool that combines text analysis
with a lightweight personality questionnaire to provide meaningful self-reflection
guidance without overstepping into clinical territory.
"""

from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import PersonalityAnalyzer


# ============================================================================
# CONFIGURATION
# ============================================================================

QUESTIONNAIRE_ITEMS = [
    {
        "id": "q_energy",
        "dimension": "IE",
        "question": "After a draining week, what restores you most?",
        "left": "Quiet time alone",
        "right": "Time with people",
    },
    {
        "id": "q_processing",
        "dimension": "IE",
        "question": "When thinking through something difficult, you usually...",
        "left": "Process internally first",
        "right": "Think out loud",
    },
    {
        "id": "q_patterns",
        "dimension": "NS",
        "question": "When learning something new, what pulls you in first?",
        "left": "Patterns and possibilities",
        "right": "Concrete facts",
    },
    {
        "id": "q_decisions",
        "dimension": "TF",
        "question": "In decisions, what weighs more?",
        "left": "Logic and consistency",
        "right": "Values and impact",
    },
    {
        "id": "q_structure",
        "dimension": "JP",
        "question": "Your best days usually feel...",
        "left": "Structured and planned",
        "right": "Flexible and open",
    },
    {
        "id": "q_deadlines",
        "dimension": "JP",
        "question": "When a deadline is near, you prefer to...",
        "left": "Lock a plan",
        "right": "Adapt as you go",
    },
]

ANSWER_TO_SCORE = {"Left": 0.0, "In Between": 0.5, "Right": 1.0}


# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="Cognisight - Journaling Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --panel: rgba(18, 26, 44, 0.88);
            --border: rgba(150, 164, 207, 0.14);
            --text: #edf2ff;
            --muted: #a7b1d1;
            --blue: #8bb6ff;
            --purple: #c7a0ff;
            --green: #94e6c1;
            --orange: #ffb177;
        }

        .stApp {
            background: radial-gradient(circle at top left, rgba(139, 182, 255, 0.15), transparent 28%),
                        radial-gradient(circle at top right, rgba(199, 160, 255, 0.12), transparent 24%),
                        linear-gradient(180deg, #080d18 0%, #0c1221 100%);
        }

        .block-container {
            max-width: 1200px;
        }

        .hero, .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 16px;
        }

        .hero {
            padding: 32px;
            background: linear-gradient(135deg, rgba(139, 182, 255, 0.12), transparent 38%),
                        linear-gradient(180deg, rgba(24, 33, 57, 0.97), rgba(15, 21, 38, 0.97));
        }

        .title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #edf2ff;
            margin: 0;
        }

        .subtitle {
            color: #a7b1d1;
            font-size: 1rem;
            margin-top: 8px;
        }

        .section-label {
            text-transform: uppercase;
            font-size: 0.75rem;
            color: #94e6c1;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #edf2ff;
            margin: 0;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 800;
            color: #edf2ff;
        }

        .stat-label {
            color: #a7b1d1;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .insight-item {
            background: rgba(139, 182, 255, 0.08);
            border-left: 3px solid #8bb6ff;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            color: #edf2ff;
            line-height: 1.6;
        }

        .strength-item {
            background: rgba(148, 230, 193, 0.08);
            border-left: 3px solid #94e6c1;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            color: #edf2ff;
        }

        .growth-item {
            background: rgba(255, 177, 119, 0.08);
            border-left: 3px solid #ffb177;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            color: #edf2ff;
        }

        .prompt-item {
            background: rgba(199, 160, 255, 0.08);
            border-left: 3px solid #c7a0ff;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            color: #edf2ff;
            font-style: italic;
        }

        .safe-mode-box {
            background: rgba(255, 177, 119, 0.1);
            border: 1px solid rgba(255, 177, 119, 0.3);
            border-radius: 12px;
            padding: 20px;
            color: #edf2ff;
        }

        .stTextArea textarea {
            background: rgba(16, 23, 40, 0.96) !important;
            color: #edf2ff !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        .stButton button {
            background: linear-gradient(90deg, #8bb6ff, #c7a0ff) !important;
            color: #09101d !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 10px 20px !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(18, 26, 44, 0.72);
            border: 1px solid var(--border);
            border-radius: 999px;
            color: #a7b1d1;
            padding: 8px 16px;
        }

        .disclaimer {
            background: rgba(139, 182, 255, 0.06);
            border: 1px solid rgba(139, 182, 255, 0.12);
            border-radius: 12px;
            padding: 12px 14px;
            font-size: 0.85rem;
            color: #a7b1d1;
            margin-top: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_analyzer() -> PersonalityAnalyzer:
    """Load analyzer once and cache across reruns."""
    return PersonalityAnalyzer()


def state_to_color(state: str) -> str:
    """Map mental state to color."""
    colors = {
        "Calm": "#94e6c1",
        "Reflective": "#c7a0ff",
        "Analytical": "#8bb6ff",
        "Stressed": "#ffb177",
        "Overthinking": "#ff9ed1",
        "Mixed": "#a7b1d1",
        "Unclear": "#a7b1d1",
    }
    return colors.get(state, "#8bb6ff")


def render_hero():
    """Render header/hero section."""
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">AI-Powered Journaling</div>
            <div class="title">Cognisight</div>
            <div class="subtitle">
                Combining text analysis with a brief personality check-in to turn journaling into actionable self-reflection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_area() -> str:
    """Render entry text input."""
    st.markdown(
        """
        <div class="card">
            <div class="section-label">📝 Your Journal Entry</div>
            <div class="section-title">Write Your Thoughts</div>
            <div style="color: #a7b1d1; font-size: 0.9rem; margin-top: 4px;">
                Share what's on your mind, what you're feeling, or what you're trying to work through. 
                The more authentic, the better the insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Journal entry",
        label_visibility="collapsed",
        placeholder="Write about what happened, what you're feeling, what's on your mind...",
        height=200,
    )
    return text


def render_questionnaire() -> Dict[str, float]:
    """Render personality questionnaire."""
    st.markdown(
        """
        <div class="card">
            <div class="section-label">🎯 Quick Check-In</div>
            <div class="section-title">6 Quick Questions</div>
            <div style="color: #a7b1d1; font-size: 0.9rem; margin-top: 4px;">
                These help personalize the analysis when text alone is ambiguous. Your answers stay private.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dimension_scores: Dict[str, List[float]] = {}
    cols = st.columns(2)

    for idx, item in enumerate(QUESTIONNAIRE_ITEMS):
        with cols[idx % 2]:
            response = st.radio(
                item["question"],
                ["Left", "In Between", "Right"],
                horizontal=True,
                key=item["id"],
                format_func=lambda x, left=item["left"], right=item["right"]: {
                    "Left": left,
                    "In Between": "Middle",
                    "Right": right,
                }[x],
            )
            dim = item["dimension"]
            dimension_scores.setdefault(dim, []).append(ANSWER_TO_SCORE[response])

    return {
        dim: sum(scores) / len(scores)
        for dim, scores in dimension_scores.items()
    }


def render_overview_results(results: Dict):
    """Render main results overview tab."""
    col1, col2, col3 = st.columns(3)

    # Mental State
    with col1:
        state = results["mental_state"]["label"]
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-label">Mental State</div>
                <div class="stat-value" style="color: {state_to_color(state)};">{state}</div>
                <div style="color: #c7a0ff; font-size: 0.85rem; margin-top: 8px;">
                    {results["mental_state"]["summary"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MBTI Primary
    with col2:
        primary = results["mbti_primary"]
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-label">MBTI Fit</div>
                <div class="stat-value">{primary["type"]}</div>
                <div style="color: #a7b1d1; font-size: 0.8rem; margin-top: 8px;">
                    Confidence: {primary["probability"]:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Self-Awareness
    with col3:
        sa = results["self_awareness"]
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-label">Self-Awareness</div>
                <div class="stat-value">{sa["score"]}/100</div>
                <div style="color: #8bb6ff; font-size: 0.8rem; margin-top: 8px;">
                    {sa["label"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Session memory message
    if results.get("memory_message"):
        st.markdown(
            f"""
            <div class="disclaimer">
                {results["memory_message"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Emotional Analysis
    st.markdown(
        """
        <div class="card">
            <div class="section-label">❤️ Emotional Signature</div>
            <div class="section-title">How You're Feeling</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for insight in results["emotional_analysis"]["insights"]:
        st.markdown(
            f'<div class="insight-item">{insight}</div>',
            unsafe_allow_html=True,
        )


def render_insights_tab(results: Dict):
    """Render detailed insights tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">🔄 Thought Patterns</div>
                <div class="section-title">How Your Mind Is Working</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for pattern in results["thought_patterns"]:
            st.markdown(
                f'<div class="insight-item">{pattern}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">⚠️ Mental Signals</div>
                <div class="section-title">What the Language Shows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for signal in results["mental_signals"]:
            st.markdown(
                f'<div class="insight-item">{signal}</div>',
                unsafe_allow_html=True,
            )

    # Communication Style
    st.markdown(
        """
        <div class="card">
            <div class="section-label">🗣️ Communication Style</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for style in results["communication_style"]:
        st.markdown(
            f'<div class="insight-item">{style}</div>',
            unsafe_allow_html=True,
        )

    # Timeline visualization
    if results.get("timeline"):
        st.markdown(
            """
            <div class="card">
                <div class="section-label">📈 Emotional Journey</div>
                <div class="section-title">Sentiment Through Your Entry</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        timeline_df = pd.DataFrame(results["timeline"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timeline_df["step"],
                y=timeline_df["sentiment"],
                mode="lines+markers",
                line=dict(color="#8bb6ff", width=3),
                marker=dict(size=8, color="#c7a0ff"),
                hovertemplate="Sentence %{x}<br>Sentiment: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=280,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a7b1d1", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", range=[-1, 1]),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_personality_tab(results: Dict):
    """Render personality type tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">🎭 MBTI Spectrum</div>
                <div class="section-title">Top Type Matches</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for match in results["mbti_matches"]:
            st.markdown(
                f'<div class="insight-item"><strong>{match["type"]}</strong> — {match["probability"]:.0f}%</div>',
                unsafe_allow_html=True,
            )

        # Type explanations
        st.markdown(
            '<div style="background: rgba(139, 182, 255, 0.06); border-radius: 8px; padding: 12px; margin-top: 12px;">',
            unsafe_allow_html=True,
        )
        for explanation in results["mbti_insights"]:
            st.markdown(f'• {explanation}')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">🔍 Word Highlights</div>
                <div class="section-title">What Influenced the Reading</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(results["highlighted_text_html"], unsafe_allow_html=True)
        st.caption("\n".join(results["highlight_legend"]))


def render_growth_tab(results: Dict):
    """Render growth and next steps tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">✨ Strengths</div>
                <div class="section-title">What You're Doing Well</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for strength in results["strengths"]:
            st.markdown(
                f'<div class="strength-item">{strength}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">🌱 Next Steps</div>
                <div class="section-title">Growth Opportunities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for suggestion in results["suggestions"]:
            st.markdown(
                f'<div class="growth-item">{suggestion}</div>',
                unsafe_allow_html=True,
            )

    # Reflection Prompts
    if results.get("reflection_prompts"):
        st.markdown(
            """
            <div class="card">
                <div class="section-label">💭 Continue Reflecting</div>
                <div class="section-title">Questions for Your Next Entry</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for prompt in results["reflection_prompts"]:
            st.markdown(
                f'<div class="prompt-item">{prompt}</div>',
                unsafe_allow_html=True,
            )


def render_comparison_section(analyzer: PersonalityAnalyzer):
    """Render comparison feature."""
    with st.expander("📊 Compare Two Entries", expanded=False):
        st.markdown(
            "Compare how your thinking, mood, and overall state changed between two different journal entries."
        )

        col1, col2 = st.columns(2)
        with col1:
            text1 = st.text_area(
                "Earlier entry",
                key="cmp_text1",
                label_visibility="collapsed",
                placeholder="An earlier entry...",
                height=150,
            )
        with col2:
            text2 = st.text_area(
                "Later entry",
                key="cmp_text2",
                label_visibility="collapsed",
                placeholder="A more recent entry...",
                height=150,
            )

        if st.button("⚖️ Compare & Analyze", use_container_width=True):
            if not text1.strip() or not text2.strip():
                st.error("Please provide both entries.")
                return

            with st.spinner("Comparing entries..."):
                comparison = analyzer.compare_texts(text1, text2)

            if not comparison.get("success"):
                st.error(comparison.get("error", "Comparison failed."))
                return

            if comparison.get("safe_mode"):
                st.warning("One entry triggered the safety layer. Comparison paused.")
                return

            if comparison.get("low_signal"):
                st.info("One entry is too low-signal for comparison.")
                return

            # Show comparison results
            st.markdown("### Comparison Summary")
            st.markdown(comparison["comparison_summary"])

            cols = st.columns(len(comparison["shift_cards"]))
            for col, card in zip(cols, comparison["shift_cards"]):
                with col:
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="stat-label">{card["label"]}</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #8bb6ff; margin: 8px 0;">
                                {card["after"]}
                            </div>
                            <div style="font-size: 0.8rem; color: #a7b1d1;">Was: {card["before"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Trait changes
            if comparison.get("differences"):
                st.subheader("Trait shifts")
                df_diffs = pd.DataFrame(comparison["differences"])
                st.dataframe(df_diffs, use_container_width=True, hide_index=True)


def render_safe_mode(results: Dict):
    """Render safe mode warning."""
    st.markdown(
        f"""
        <div class="safe-mode-box">
            <h3 style="color: #ffb177; margin-top: 0;">⚠️ Supportive Check-In</h3>
            <p>{results["support_message"]}</p>
            <p><strong>Resources:</strong> If you're in crisis, please reach out to a counselor or trusted person.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_low_signal(results: Dict):
    """Render low signal warning."""
    st.markdown(
        """
        <div class="card">
            <div class="section-label">📌 Low Signal Entry</div>
            <div class="section-title">More Detail Needed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**{results['reflection_summary']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Why this isn't enough:**")
        for pattern in results["thought_patterns"]:
            st.markdown(f'<div class="insight-item">{pattern}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Try this next time:**")
        for suggestion in results["suggestions"]:
            st.markdown(f'<div class="growth-item">{suggestion}</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP LOGIC
# ============================================================================

def main():
    """Main application flow."""
    analyzer = load_analyzer()
    render_hero()

    # Input layout
    col_input, col_q = st.columns([1.3, 1.0])

    with col_input:
        text = render_input_area()

    with col_q:
        questionnaire = render_questionnaire()

    # Analyze button
    col1, col2 = st.columns([0.35, 0.65])
    with col1:
        analyze_button = st.button("🔍 Analyze", use_container_width=True)
    with col2:
        st.caption("The questionnaire helps clarify MBTI fit when text alone is ambiguous.")

    if analyze_button:
        if not text.strip():
            st.error("Write something before analyzing.")
            return

        # Load previous result for session memory
        previous = st.session_state.get("last_result")

        with st.spinner("Analyzing your thoughts..."):
            result = analyzer.analyze(text, questionnaire=questionnaire)

        if result.get("success"):
            if not result.get("safe_mode") and not result.get("low_signal"):
                # Add session memory message if there's a previous result
                if previous:
                    before = previous["self_awareness"]["score"]
                    after = result["self_awareness"]["score"]
                    before_state = previous["mental_state"]["label"]
                    after_state = result["mental_state"]["label"]

                    if after >= before + 8:
                        result["memory_message"] = f"📈 You're clearer today ({before} → {after}/100)"
                    elif after <= before - 8:
                        result["memory_message"] = f"📉 Today feels less settled ({before} → {after}/100)"
                    elif before_state != after_state:
                        result["memory_message"] = f"State shift: {before_state} → {after_state}"

                # Update session history
                st.session_state.last_result = result

            st.session_state.analysis_result = result

    # Display results
    result = st.session_state.get("analysis_result")
    if result:
        if not result.get("success"):
            st.error(f"Analysis error: {result.get('error')}")
        elif result.get("safe_mode"):
            render_safe_mode(result)
        elif result.get("low_signal"):
            render_low_signal(result)
        else:
            # Main results with tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Insights", "Personality", "Growth"])

            with tab1:
                render_overview_results(result)

            with tab2:
                render_insights_tab(result)

            with tab3:
                render_personality_tab(result)

            with tab4:
                render_growth_tab(result)

            # Disclaimer
            st.markdown(
                f"""
                <div class="disclaimer">
                    {result["disclaimer"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Comparison feature
    render_comparison_section(analyzer)


if __name__ == "__main__":
    main()
