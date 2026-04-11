"""
Cognisight Streamlit application.
"""

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import PersonalityAnalyzer


QUESTIONNAIRE_ITEMS = [
    {
        "id": "q_energy",
        "dimension": "IE",
        "question": "After a draining week, what restores you more?",
        "left": "Quiet time alone",
        "right": "Time with other people",
    },
    {
        "id": "q_processing",
        "dimension": "IE",
        "question": "When you're thinking through something difficult, you usually...",
        "left": "Process internally first",
        "right": "Think out loud with others",
    },
    {
        "id": "q_patterns",
        "dimension": "NS",
        "question": "When learning something new, what pulls you in first?",
        "left": "Patterns and possibilities",
        "right": "Facts and concrete details",
    },
    {
        "id": "q_decisions",
        "dimension": "TF",
        "question": "What weighs more in decisions?",
        "left": "Logic and consistency",
        "right": "Values and people impact",
    },
    {
        "id": "q_structure",
        "dimension": "JP",
        "question": "Your best days usually feel...",
        "left": "Mapped out and structured",
        "right": "Flexible and open-ended",
    },
    {
        "id": "q_deadlines",
        "dimension": "JP",
        "question": "When a deadline is close, you prefer to...",
        "left": "Lock a plan and follow it",
        "right": "Adapt as you go",
    },
]

ANSWER_TO_SCORE = {
    "Left": 0.0,
    "In Between": 0.5,
    "Right": 1.0,
}


st.set_page_config(
    page_title="Cognisight",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        :root {
            --panel: rgba(18, 26, 44, 0.88);
            --panel-strong: rgba(22, 31, 54, 0.96);
            --border: rgba(150, 164, 207, 0.14);
            --text: #edf2ff;
            --muted: #a7b1d1;
            --blue: #8bb6ff;
            --purple: #c7a0ff;
            --pink: #ff9ed1;
            --amber: #f6cf7a;
            --orange: #ffb177;
            --green: #94e6c1;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(139, 182, 255, 0.15), transparent 28%),
                radial-gradient(circle at top right, rgba(199, 160, 255, 0.12), transparent 24%),
                linear-gradient(180deg, #080d18 0%, #0c1221 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero, .card, .stat, .soft-note {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
            animation: fadeUp 0.45s ease both;
            transition: transform 0.22s ease, border-color 0.22s ease;
        }

        .card:hover, .stat:hover {
            transform: translateY(-2px);
            border-color: rgba(199, 160, 255, 0.28);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .hero {
            padding: 32px 34px;
            margin-bottom: 1rem;
            background:
                linear-gradient(135deg, rgba(139, 182, 255, 0.12), transparent 38%),
                linear-gradient(180deg, rgba(24, 33, 57, 0.97), rgba(15, 21, 38, 0.97));
        }

        .card {
            padding: 20px 22px;
            margin-bottom: 1rem;
            background: var(--panel-strong);
        }

        .stat {
            padding: 18px 20px;
            min-height: 124px;
        }

        .soft-note {
            padding: 14px 16px;
            margin-top: 0.8rem;
            background: rgba(139, 182, 255, 0.08);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            color: var(--green);
            font-weight: 700;
        }

        .title {
            color: var(--text);
            font-size: 3rem;
            line-height: 1;
            font-weight: 800;
            margin: 0.3rem 0 0.8rem 0;
        }

        .muted {
            color: var(--muted);
            font-size: 0.98rem;
        }

        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.74rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }

        .stat-value {
            color: var(--text);
            font-size: 1.85rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .stat-subtle {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.34rem 0.82rem;
            margin-right: 0.45rem;
            font-size: 0.84rem;
            font-weight: 700;
        }

        .progress-shell {
            margin-top: 0.65rem;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.35rem;
        }

        .progress-bar {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
        }

        .match-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }

        .match-name {
            color: var(--text);
            font-weight: 700;
            min-width: 56px;
        }

        .match-bar {
            height: 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
            flex: 1;
        }

        .match-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--purple), var(--blue));
        }

        .signal-box, .highlight-box {
            background: rgba(139, 182, 255, 0.06);
            border: 1px solid rgba(139, 182, 255, 0.08);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 0.75rem;
        }

        .highlight-box {
            color: var(--text);
            line-height: 1.8;
        }

        .stTextArea textarea {
            background: rgba(16, 23, 40, 0.96) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            min-height: 260px !important;
            padding: 1rem !important;
        }

        .stButton button {
            background: linear-gradient(90deg, #8bb6ff, #c7a0ff) !important;
            color: #09101d !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.85rem 1.3rem !important;
            font-weight: 800 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(18, 26, 44, 0.72);
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--text);
            padding: 0.48rem 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_analyzer() -> PersonalityAnalyzer:
    return PersonalityAnalyzer()


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Hybrid Journaling Intelligence</div>
            <div class="title">Cognisight</div>
            <div class="muted">
                Cognisight combines text analysis with a tiny personality questionnaire to create a
                more personalized, trustworthy journaling assistant. It reads your writing, checks
                your preferences, and turns both into calm, useful guidance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_area() -> str:
    with st.container():
        st.markdown(
            """
            <div class="card">
                <div class="section-label">Write Your Thoughts</div>
                <div class="section-title">Journal Entry</div>
                <div class="muted">
                    This tool is for self-reflection only and not a substitute for professional mental health support.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return st.text_area(
            "Write your thoughts",
            label_visibility="collapsed",
            placeholder="Write about what happened, what you are feeling, what keeps replaying in your mind, or what you are trying to figure out...",
        )


def render_questionnaire() -> Dict[str, float]:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Micro-Questionnaire</div>
            <div class="section-title">Quick personality check-in</div>
            <div class="muted">Six quick prompts help the app personalize MBTI-style matches when text alone is ambiguous.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dimension_scores: Dict[str, List[float]] = {}
    col1, col2 = st.columns(2)
    columns = [col1, col2]

    for index, item in enumerate(QUESTIONNAIRE_ITEMS):
        with columns[index % 2]:
            selection = st.radio(
                item["question"],
                ["Left", "In Between", "Right"],
                horizontal=True,
                key=item["id"],
                format_func=lambda key, left=item["left"], right=item["right"]: {
                    "Left": left,
                    "In Between": "In between",
                    "Right": right,
                }[key],
            )
            dimension_scores.setdefault(item["dimension"], []).append(ANSWER_TO_SCORE[selection])

    return {
        dimension: sum(values) / len(values)
        for dimension, values in dimension_scores.items()
    }


def state_color(label: str) -> str:
    mapping = {
        "Calm": "#8bb6ff",
        "Reflective": "#c7a0ff",
        "Analytical": "#94e6c1",
        "Stressed": "#ffb177",
        "Overthinking": "#ff9ed1",
        "Mixed": "#a7b1d1",
        "Unclear": "#a7b1d1",
    }
    return mapping.get(label, "#8bb6ff")


def render_progress_metric(title: str, value: int, color: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="section-label">{title}</div>
            <div class="stat-value">{value}/100</div>
            <div class="progress-shell">
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{value}%; background:{color};"></div>
                </div>
            </div>
            {f"<div class='stat-subtle' style='margin-top:0.55rem;'>{subtitle}</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list_card(label: str, title: str, items) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="section-label">{label}</div>
            <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )
    for item in items:
        st.markdown(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_safe_message(results) -> None:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Supportive Response</div>
            <div class="section-title">Analysis paused</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(results["support_message"])
    st.markdown(
        f"<div class='soft-note'><span class='muted'>{results['disclaimer']}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_low_signal(results) -> None:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Low-Signal Entry</div>
            <div class="section-title">The sample needs more context</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(results["reflection_summary"])
    st.markdown("</div>", unsafe_allow_html=True)
    render_list_card("Thought Patterns", "Why the result would be unreliable", results["thought_patterns"])
    render_list_card("Suggestions", "How to make the next entry more useful", results["suggestions"])


def render_memory_message(message: str) -> None:
    if not message:
        return
    st.markdown(
        f"""
        <div class="soft-note">
            <span class="muted">{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_memory_message(previous: Dict, current: Dict) -> str:
    if not previous:
        return ""

    before_score = previous["self_awareness"]["score"]
    after_score = current["self_awareness"]["score"]
    before_state = previous["mental_state"]["label"]
    after_state = current["mental_state"]["label"]

    if after_score >= before_score + 8:
        return f"Compared to your last entry, you seem more clear and self-aware. The score moved from {before_score} to {after_score}."
    if after_score <= before_score - 8:
        return f"Compared to your last entry, this one feels less settled. The self-awareness score moved from {before_score} to {after_score}."
    if before_state != after_state:
        return f"Compared to your last entry, your mental state shifted from {before_state.lower()} toward {after_state.lower()}."
    return "Compared to your last entry, the overall pattern is fairly similar, even if the tone and details changed."


def render_mbti_matches(results) -> None:
    rows = []
    for match in results["mbti_matches"]:
        rows.append(
            f"""
            <div class="match-row">
                <div class="match-name">{match['type']}</div>
                <div class="match-bar"><div class="match-fill" style="width:{match['probability']:.0f}%"></div></div>
                <div class="stat-subtle">{match['probability']:.0f}%</div>
            </div>
            """
        )
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Primary Type</div>
            <div class="section-title">"""
        + f"{results['mbti_primary']['type']} (Confidence: {results['mbti_primary']['probability']:.0f}%)"
        + """</div>
        """
        + "".join(rows),
        unsafe_allow_html=True,
    )
    for line in results["mbti_insights"]:
        st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_questionnaire_summary(results) -> None:
    summary = results["fusion"]["questionnaire_dimensions"]
    if not summary:
        st.info("No questionnaire data was used in this analysis.")
        return

    st.markdown(
        """
        <div class="card">
            <div class="section-label">Questionnaire Blend</div>
            <div class="section-title">How the quick check-in influenced the type fit</div>
        """,
        unsafe_allow_html=True,
    )
    for row in summary:
        st.markdown(
            f"""
            <div class="progress-shell">
                <div class="progress-label">
                    <span>{row['left']} ↔ {row['right']}</span>
                    <span>{row['lean']}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{row['score']*100:.0f}%; background: linear-gradient(90deg, #8bb6ff, #c7a0ff);"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='soft-note'><span class='muted'>Text weight: {results['fusion']['text_weight']:.0%} • Questionnaire weight: {results['fusion']['questionnaire_weight']:.0%}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_timeline(results) -> None:
    timeline_df = pd.DataFrame(results["timeline"])
    if timeline_df.empty:
        st.info("No sentence-level timeline available.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline_df["step"],
            y=timeline_df["sentiment"],
            mode="lines+markers",
            line=dict(color="#8bb6ff", width=3),
            marker=dict(size=9, color="#c7a0ff"),
            text=timeline_df["sentence"],
            hovertemplate="Sentence %{x}<br>Sentiment %{y:.2f}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color="#edf2ff"),
        yaxis=dict(range=[-1, 1], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Sentence"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher values are more positive, lower values are heavier or more negative.")


def render_highlighted_text(results) -> None:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Highlight Insights</div>
            <div class="section-title">Words and phrases shaping the analysis</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='highlight-box'>{results['highlighted_text_html']}</div>", unsafe_allow_html=True)
    for line in results["highlight_legend"]:
        st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_key_signals(results) -> None:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Key Signals</div>
            <div class="section-title">Language patterns influencing the output</div>
        """,
        unsafe_allow_html=True,
    )
    for feature_name, value, explanation in results["key_signals"]:
        st.markdown(
            f"""
            <div class="signal-box">
                <strong>{feature_name}</strong> • {value:.2f}<br>
                <span class="muted">{explanation}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_overview_tab(results) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="stat">
                <div class="stat-label">Mental State</div>
                <div class="stat-value" style="color:{state_color(results['mental_state']['label'])};">{results['mental_state']['label']}</div>
                <div class="stat-subtle">{results['reflection_summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        primary = results["mbti_primary"]
        st.markdown(
            f"""
            <div class="stat">
                <div class="stat-label">Primary MBTI Fit</div>
                <div class="stat-value">{primary['type']}</div>
                <div class="stat-subtle">Confidence: {primary['probability']:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="stat">
                <div class="stat-label">Self-Awareness Score</div>
                <div class="stat-value">{results['self_awareness']['score']}/100</div>
                <div class="stat-subtle">{results['self_awareness']['label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_memory_message(results.get("memory_message", ""))
    render_list_card("Emotional Summary", "How the emotional signal looks", results["emotional_analysis"]["insights"])
    render_progress_metric(
        "Text Signal Strength",
        int(round(results["fusion"]["text_strength"] * 100)),
        "#8bb6ff",
        "Higher signal means the text carried stronger evidence on its own.",
    )


def render_insights_tab(results) -> None:
    col1, col2 = st.columns(2)
    with col1:
        render_list_card("Thought Patterns", "How the thinking flow reads", results["thought_patterns"])
        render_list_card("Mental Signals", "Non-clinical strain indicators", results["mental_signals"])
    with col2:
        render_list_card("Communication Style", "How the writing comes across", results["communication_style"])
        render_timeline(results)


def render_personality_tab(results) -> None:
    left, right = st.columns([1.05, 0.95])
    with left:
        render_mbti_matches(results)
        render_questionnaire_summary(results)
    with right:
        render_highlighted_text(results)
        render_key_signals(results)


def render_growth_tab(results) -> None:
    col1, col2 = st.columns(2)
    with col1:
        render_list_card("Strengths", "What already looks strong", results["strengths"])
        render_list_card("Growth Mode", "How you can improve", results["growth_mode"])
    with col2:
        breakdown = results["self_awareness"]["breakdown"]
        render_progress_metric("Clarity", breakdown["clarity"], "#8bb6ff")
        render_progress_metric("Emotional Stability", breakdown["emotional_stability"], "#ffb177")
        render_progress_metric("Reflection Depth", breakdown["reflection_depth"], "#c7a0ff")


def render_results(results) -> None:
    overview_tab, insights_tab, personality_tab, growth_tab = st.tabs(
        ["Overview", "Insights", "Personality", "Growth"]
    )
    with overview_tab:
        render_overview_tab(results)
    with insights_tab:
        render_insights_tab(results)
    with personality_tab:
        render_personality_tab(results)
    with growth_tab:
        render_growth_tab(results)

    st.markdown(
        f"<div class='soft-note'><span class='muted'>{results['confidence_note']} {results['disclaimer']}</span></div>",
        unsafe_allow_html=True,
    )


def render_compare_section(analyzer: PersonalityAnalyzer) -> None:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Compare Entries</div>
            <div class="section-title">Before vs after</div>
            <div class="muted">Compare two journal entries to see how your tone, structure, and overall reading changed.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area(
            "Entry A",
            key="compare_a",
            label_visibility="collapsed",
            placeholder="Earlier entry...",
        )
    with col2:
        text2 = st.text_area(
            "Entry B",
            key="compare_b",
            label_visibility="collapsed",
            placeholder="Later entry...",
        )

    if st.button("Compare Entries", use_container_width=True):
        if not text1.strip() or not text2.strip():
            st.error("Enter both texts before comparing them.")
            return

        comparison = analyzer.compare_texts(text1, text2)
        st.session_state.compare_result = comparison

    comparison = st.session_state.get("compare_result")
    if not comparison:
        return
    if not comparison.get("success"):
        st.error(comparison.get("error", "Comparison failed."))
        return
    if comparison.get("safe_mode") or comparison.get("low_signal"):
        st.info(comparison["comparison_summary"])
        return

    st.markdown(
        f"""
        <div class="card">
            <div class="section-label">Comparison Summary</div>
            <div class="section-title">{comparison['comparison_summary']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(comparison["shift_cards"]))
    for col, card in zip(cols, comparison["shift_cards"]):
        with col:
            st.markdown(
                f"""
                <div class="stat">
                    <div class="stat-label">{card['label']}</div>
                    <div class="stat-value" style="font-size:1.25rem;">{card['after']}</div>
                    <div class="stat-subtle">Before: {card['before']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.dataframe(pd.DataFrame(comparison["differences"]), use_container_width=True, hide_index=True)


def main() -> None:
    analyzer = load_analyzer()
    render_hero()

    input_col, questionnaire_col = st.columns([1.35, 1.0])
    with input_col:
        text = render_input_area()
    with questionnaire_col:
        questionnaire = render_questionnaire()

    action_col, re_col = st.columns([0.3, 0.7])
    with action_col:
        analyze_clicked = st.button("Analyze My Thoughts", use_container_width=True)
    with re_col:
        st.caption("Text usually carries most of the weight. The questionnaire helps when the entry is shorter or more ambiguous.")

    if analyze_clicked:
        if not text.strip():
            st.error("Write your thoughts before running the analysis.")
            return

        previous = st.session_state.get("last_successful_result")
        with st.spinner("Blending text signals with your quick check-in..."):
            result = analyzer.analyze(text, questionnaire=questionnaire)

        if (
            result.get("success")
            and not result.get("safe_mode")
            and not result.get("low_signal")
        ):
            result["memory_message"] = build_memory_message(previous, result) if previous else ""
            st.session_state.last_successful_result = result
            history = st.session_state.get("analysis_history", [])
            history.append(
                {
                    "mental_state": result["mental_state"],
                    "self_awareness": result["self_awareness"],
                    "mbti_primary": result["mbti_primary"],
                    "emotional_analysis": result["emotional_analysis"],
                }
            )
            st.session_state.analysis_history = history[-5:]

        st.session_state.analysis_result = result

    result = st.session_state.get("analysis_result")
    if result:
        if not result.get("success"):
            st.error(result.get("error", "Analysis failed."))
        elif result.get("safe_mode"):
            render_safe_message(result)
        elif result.get("low_signal"):
            render_low_signal(result)
        else:
            render_results(result)

    render_compare_section(analyzer)


if __name__ == "__main__":
    main()
