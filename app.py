"""
Cognisight — Mental Health Journal & Personality Analyzer
User accounts · Daily check-ins · Personality onboarding · Trend tracking
"""

import json
import os
import hashlib
import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import PersonalityAnalyzer


# ============================================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================================

st.set_page_config(
    page_title="Cognisight",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# GLOBAL STYLES
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0b0f1a;
    --surface:   #131929;
    --surface2:  #1a2235;
    --border:    rgba(130,155,210,0.13);
    --border2:   rgba(130,155,210,0.22);
    --text:      #e8edf8;
    --muted:     #8a96b8;
    --accent:    #7c9ef8;
    --accent2:   #b48bff;
    --green:     #6edcaa;
    --orange:    #f5a55a;
    --red:       #f07070;
    --radius:    14px;
    --radius-sm: 8px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1100px; padding: 2rem 2rem 4rem; }

/* Remove streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 14px;
}
.card-sm {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* Typography */
.display { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: var(--text); line-height: 1.15; }
.display em { font-style: italic; color: var(--accent); }
.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.section-title { font-size: 1.2rem; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.muted { color: var(--muted); font-size: 0.9rem; }

/* Stat blocks */
.stat-label { font-size: 0.75rem; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--text); }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: opacity .15s !important;
}
.stButton button:hover { opacity: .88 !important; }
.stButton button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,158,248,0.15) !important;
}

/* Selectbox / radio */
.stRadio label { color: var(--text) !important; font-size: 0.9rem !important; }
.stRadio [data-baseweb="radio"] { gap: 8px !important; }
.stSelectbox select { background: var(--surface2) !important; color: var(--text) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 18px !important;
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; border-radius: 99px !important; }

/* Insight items */
.insight-item { background: var(--surface2); border-left: 3px solid var(--accent); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: 10px 14px; margin: 8px 0; font-size: 0.9rem; color: var(--text); line-height: 1.55; }
.growth-item  { background: var(--surface2); border-left: 3px solid var(--green);  border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: 10px 14px; margin: 8px 0; font-size: 0.9rem; color: var(--text); line-height: 1.55; }
.prompt-item  { background: var(--surface2); border-left: 3px solid var(--accent2); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: 10px 14px; margin: 8px 0; font-size: 0.9rem; color: var(--text); line-height: 1.55; }

/* Badges */
.badge { display: inline-block; font-size: 0.7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; padding: 3px 9px; border-radius: 99px; }
.badge-blue   { background: rgba(124,158,248,.18); color: var(--accent); }
.badge-purple { background: rgba(180,139,255,.18); color: var(--accent2); }
.badge-green  { background: rgba(110,220,170,.18); color: var(--green); }
.badge-orange { background: rgba(245,165,90,.18);  color: var(--orange); }

/* Disclaimer */
.disclaimer { background: rgba(124,158,248,.06); border: 1px solid rgba(124,158,248,.12); border-radius: var(--radius-sm); padding: 12px 16px; font-size: 0.82rem; color: var(--muted); margin-top: 20px; line-height: 1.6; }

/* Mood pill */
.mood-pill { display: inline-block; padding: 5px 14px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }

/* Divider */
.divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# USER DATA STORAGE  (file-based — swap for a DB in production)
# ============================================================================

USERS_FILE = "cognisight_users.json"
ENTRIES_FILE = "cognisight_entries.json"


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def get_users() -> dict:
    return _load_json(USERS_FILE)


def save_user(username: str, data: dict):
    users = get_users()
    users[username] = data
    _save_json(USERS_FILE, users)


def get_entries(username: str) -> list:
    all_entries = _load_json(ENTRIES_FILE)
    return all_entries.get(username, [])


def save_entry(username: str, entry: dict):
    all_entries = _load_json(ENTRIES_FILE)
    if username not in all_entries:
        all_entries[username] = []
    all_entries[username].append(entry)
    _save_json(ENTRIES_FILE, all_entries)


# ============================================================================
# MODEL LOADING — explicit two-step so users see progress
# ============================================================================

@st.cache_resource(show_spinner=False)
def _warmup_emotion_models():
    """
    Load HuggingFace transformer pipelines once.
    This is separated from load_analyzer() so the spinner is visible
    and the main analyzer construction doesn't silently block.
    """
    from src.emotion_extractor import EmotionExtractor
    # Instantiating triggers the class-level singleton download
    EmotionExtractor()
    return True


@st.cache_resource(show_spinner=False)
def load_analyzer() -> PersonalityAnalyzer:
    return PersonalityAnalyzer(model_path="models/best_model.pkl")


def boot_models():
    """Call at startup. Shows a visible loading bar so the page is never blank."""
    if st.session_state.get("_models_ready"):
        return

    with st.spinner("⏳ Loading AI models — takes a moment on first run..."):
        _warmup_emotion_models()
        load_analyzer()

    st.session_state["_models_ready"] = True


# ============================================================================
# QUESTIONNAIRE DATA
# ============================================================================

ONBOARDING_QUESTIONS = [
    {"id": "ob_energy",     "dim": "IE", "q": "After a long social event, you feel…",          "left": "Drained — need alone time",   "right": "Energised — want more"},
    {"id": "ob_thinking",   "dim": "NS", "q": "When solving a problem, you prefer…",           "left": "Intuition & possibilities",   "right": "Concrete facts & steps"},
    {"id": "ob_decisions",  "dim": "TF", "q": "You make decisions based mostly on…",           "left": "Logic and fairness",          "right": "Values and feelings"},
    {"id": "ob_planning",   "dim": "JP", "q": "Your ideal week looks like…",                   "left": "Planned & structured",        "right": "Flexible & spontaneous"},
    {"id": "ob_conflict",   "dim": "TF", "q": "In conflict, you tend to…",                     "left": "Argue logically",             "right": "Prioritise harmony"},
    {"id": "ob_novelty",    "dim": "NS", "q": "You're drawn more to…",                         "left": "Abstract ideas & theories",   "right": "Practical how-tos"},
    {"id": "ob_social",     "dim": "IE", "q": "In groups, you're usually…",                    "left": "A quiet observer",            "right": "An active participant"},
    {"id": "ob_deadlines",  "dim": "JP", "q": "With deadlines you…",                           "left": "Plan ahead to avoid rush",    "right": "Work well under pressure"},
]

DAILY_MOOD_OPTIONS = ["😄 Great", "🙂 Good", "😐 Okay", "😔 Low", "😰 Anxious", "😤 Frustrated", "😴 Exhausted"]

DAILY_QUESTIONS = [
    "How are you feeling right now, and what's sitting heaviest on your mind?",
    "What happened today that affected your mood the most?",
    "What are you worried or uncertain about? What would make it easier?",
    "What did you do for yourself today, even something small?",
    "What's one thing you need that you haven't been getting?",
    "Describe a moment today where you felt like yourself — or the opposite.",
    "What's taking up the most mental energy right now?",
]

ANSWER_TO_SCORE = {"Left": 0.0, "Middle": 0.5, "Right": 1.0}


# ============================================================================
# AUTH SCREENS
# ============================================================================

def render_auth():
    """Login / register gate."""
    st.markdown("""
    <div style="max-width:460px; margin: 4rem auto 0;">
        <div class="display" style="text-align:center; margin-bottom: 6px;">
            Cogni<em>sight</em>
        </div>
        <p style="text-align:center; color:var(--muted); margin-bottom: 2rem;">
            Your private mental health journal &amp; personality tracker
        </p>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        tab_login, tab_reg = st.tabs(["Sign in", "Create account"])

        with tab_login:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="your username")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
            if st.button("Sign in", use_container_width=True, key="btn_login"):
                users = get_users()
                if username in users and users[username]["password"] == hash_password(password):
                    st.session_state.username = username
                    st.session_state.user_data = users[username]
                    st.rerun()
                else:
                    st.error("Username or password is wrong.")

        with tab_reg:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            new_user = st.text_input("Choose a username", key="reg_user")
            new_name = st.text_input("Your first name", key="reg_name")
            new_pw   = st.text_input("Password", type="password", key="reg_pw")
            new_pw2  = st.text_input("Confirm password", type="password", key="reg_pw2")
            if st.button("Create account", use_container_width=True, key="btn_reg"):
                users = get_users()
                if not new_user or not new_pw:
                    st.error("Username and password are required.")
                elif new_user in users:
                    st.error("That username is taken.")
                elif new_pw != new_pw2:
                    st.error("Passwords don't match.")
                else:
                    save_user(new_user, {
                        "password": hash_password(new_pw),
                        "name": new_name or new_user,
                        "onboarding_done": False,
                        "onboarding_scores": {},
                        "created_at": str(datetime.datetime.now()),
                    })
                    st.session_state.username = new_user
                    st.session_state.user_data = get_users()[new_user]
                    st.rerun()


# ============================================================================
# ONBOARDING — personality baseline questions
# ============================================================================

def render_onboarding():
    user_data = st.session_state.user_data
    name = user_data.get("name", "there")

    st.markdown(f"""
    <div class="card" style="max-width:700px; margin: 2rem auto;">
        <div class="section-label">Getting started</div>
        <div class="display" style="font-size:2rem; margin-bottom: 8px;">
            Hi {name}, let's build your personality baseline
        </div>
        <p class="muted">8 quick questions help us personalise your insights from day one. Takes about 2 minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    dimension_scores: Dict[str, List[float]] = {}
    answered = 0

    for i, item in enumerate(ONBOARDING_QUESTIONS):
        st.markdown(f"""
        <div class="card-sm" style="max-width:700px; margin: 0 auto 10px;">
            <div class="section-label">Question {i+1} of {len(ONBOARDING_QUESTIONS)}</div>
            <div style="font-size: 1rem; font-weight: 500; color: var(--text); margin-bottom:10px;">{item['q']}</div>
        </div>
        """, unsafe_allow_html=True)

        resp = st.radio(
            label=item["q"],
            options=["Left", "Middle", "Right"],
            horizontal=True,
            key=f"ob_{item['id']}",
            label_visibility="collapsed",
            format_func=lambda x, l=item["left"], r=item["right"]: {
                "Left": l, "Middle": "Somewhere in between", "Right": r
            }[x],
        )
        if resp is not None:
            answered += 1
        dimension_scores.setdefault(item["dim"], []).append(ANSWER_TO_SCORE.get(resp or "Middle", 0.5))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Complete setup →", use_container_width=True):
            scores = {dim: sum(vals) / len(vals) for dim, vals in dimension_scores.items()}
            user_data["onboarding_done"] = True
            user_data["onboarding_scores"] = scores
            save_user(st.session_state.username, user_data)
            st.session_state.user_data = user_data
            st.success("Baseline saved! Redirecting…")
            st.rerun()


# ============================================================================
# SIDEBAR — nav + user info
# ============================================================================

def render_sidebar():
    with st.sidebar:
        user_data = st.session_state.user_data
        name = user_data.get("name", st.session_state.username)

        st.markdown(f"""
        <div style="padding: 12px 0 20px;">
            <div style="font-size:1.1rem; font-weight:600; color:var(--text);">{name}</div>
            <div style="font-size:0.8rem; color:var(--muted);">@{st.session_state.username}</div>
        </div>
        """, unsafe_allow_html=True)

        entries = get_entries(st.session_state.username)
        st.markdown(f"""
        <div class="card-sm" style="margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <div><div class="stat-label">Entries</div><div style="font-size:1.5rem;font-weight:600;color:var(--accent)">{len(entries)}</div></div>
                <div><div class="stat-label">Streak</div><div style="font-size:1.5rem;font-weight:600;color:var(--green)">{_compute_streak(entries)}d</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["📝 Today's Entry", "📊 My Trends", "🧠 My Personality", "📖 Entry History"],
            label_visibility="collapsed",
            key="nav_page",
        )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        if st.button("Sign out", key="btn_signout", use_container_width=True):
            for key in ["username", "user_data", "_models_ready", "analysis_result", "last_result"]:
                st.session_state.pop(key, None)
            st.rerun()

        return page


def _compute_streak(entries: list) -> int:
    if not entries:
        return 0
    dates = sorted(set(e["date"] for e in entries), reverse=True)
    today = str(datetime.date.today())
    streak = 0
    check = today
    for d in dates:
        if d == check:
            streak += 1
            prev = datetime.date.fromisoformat(check) - datetime.timedelta(days=1)
            check = str(prev)
        else:
            break
    return streak


# ============================================================================
# PAGE: TODAY'S ENTRY
# ============================================================================

def page_todays_entry(analyzer: PersonalityAnalyzer):
    user_data = st.session_state.user_data
    name = user_data.get("name", "you")
    today = str(datetime.date.today())

    entries_today = [e for e in get_entries(st.session_state.username) if e["date"] == today]
    already_done = bool(entries_today)

    hour = datetime.datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    st.markdown(f"""
    <div class="card">
        <div class="section-label">{today}</div>
        <div class="display" style="font-size:2rem; margin-bottom: 6px;">{greeting}, {name}</div>
        <p class="muted">How are you doing today? Write freely — this is just for you.</p>
    </div>
    """, unsafe_allow_html=True)

    if already_done:
        last = entries_today[-1]
        st.markdown(f"""
        <div class="card" style="border-color: rgba(110,220,170,.25);">
            <div class="section-label">✅ Today's entry saved</div>
            <div style="font-size:0.95rem; color:var(--text); margin:8px 0;">
                Mood: <strong>{last.get('mood','—')}</strong> &nbsp;·&nbsp;
                State: <strong>{last.get('mental_state','—')}</strong> &nbsp;·&nbsp;
                Self-awareness: <strong>{last.get('self_awareness_score','—')}/100</strong>
            </div>
            <p class="muted" style="font-size:0.85rem;">You've already logged today. You can add another entry below if your day changed.</p>
        </div>
        """, unsafe_allow_html=True)

    # Daily question prompt — rotates by day-of-year
    doy = datetime.date.today().timetuple().tm_yday
    prompt = DAILY_QUESTIONS[doy % len(DAILY_QUESTIONS)]

    st.markdown(f"""
    <div class="card-sm">
        <div class="section-label">Today's reflection prompt</div>
        <div style="font-size:0.95rem; color:var(--text); font-style:italic;">"{prompt}"</div>
    </div>
    """, unsafe_allow_html=True)

    # Mood selector
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    mood = st.select_slider(
        "How's your mood right now?",
        options=DAILY_MOOD_OPTIONS,
        value=DAILY_MOOD_OPTIONS[2],
        key="mood_slider",
    )

    # Journal text
    journal_text = st.text_area(
        "Your thoughts",
        label_visibility="collapsed",
        placeholder=f"{prompt}\n\nWrite whatever comes to mind...",
        height=220,
        key="journal_text",
    )

    # Energy / stress sliders
    col1, col2 = st.columns(2)
    with col1:
        energy = st.slider("Energy level today", 1, 10, 5, key="energy_slider")
    with col2:
        stress = st.slider("Stress level today", 1, 10, 5, key="stress_slider")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col_btn, col_cap = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("🔍 Analyze & Save", use_container_width=True)
    with col_cap:
        st.caption("Your entry is analyzed locally — nothing leaves your machine.")

    if analyze_btn:
        if len((journal_text or "").strip()) < 50:
            st.error("Please write at least a couple of sentences before analyzing (minimum 50 characters).")
            return

        with st.spinner("Analyzing your entry…"):
            try:
                questionnaire = user_data.get("onboarding_scores", {})
                result = analyzer.analyze(journal_text, questionnaire=questionnaire)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        if not result.get("success"):
            st.error(f"Could not analyze: {result.get('error')}")
            return

        # Build entry record
        entry = {
            "date": today,
            "timestamp": str(datetime.datetime.now()),
            "mood": mood,
            "energy": energy,
            "stress": stress,
            "text_preview": journal_text[:120] + ("…" if len(journal_text) > 120 else ""),
            "word_count": len(journal_text.split()),
            "safe_mode": result.get("safe_mode", False),
            "low_signal": result.get("low_signal", False),
        }

        if result.get("safe_mode"):
            entry["mental_state"] = "Support needed"
            entry["self_awareness_score"] = None
        elif result.get("low_signal"):
            entry["mental_state"] = "Low signal"
            entry["self_awareness_score"] = None
        else:
            entry["mental_state"] = result.get("mental_state", {}).get("label", "—")
            entry["self_awareness_score"] = result.get("self_awareness", {}).get("score")
            entry["mbti_primary"] = result.get("mbti_primary", {}).get("type", "—")
            entry["emotional_tone"] = result.get("emotional_analysis", {}).get("tone", "—")

        save_entry(st.session_state.username, entry)
        st.session_state.analysis_result = result
        st.session_state.last_result = result

        if result.get("safe_mode"):
            _render_safe_mode(result)
        elif result.get("low_signal"):
            _render_low_signal(result)
        else:
            _render_analysis_results(result, mood, energy, stress)

    elif st.session_state.get("analysis_result"):
        result = st.session_state.analysis_result
        if not result.get("safe_mode") and not result.get("low_signal"):
            _render_analysis_results(result, None, None, None)


def _render_safe_mode(result: dict):
    st.markdown("""
    <div class="card" style="border-color: rgba(240,112,112,.3);">
        <div class="section-label" style="color:var(--red);">⚠️ We noticed something</div>
        <div class="section-title">You don't have to carry this alone</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**{result.get('support_message','')}**")
    st.info("If you're in crisis please reach out: iCall India — 9152987821 | Vandrevala Foundation — 1860-2662-345 (24/7)")


def _render_low_signal(result: dict):
    st.markdown("""
    <div class="card">
        <div class="section-label">More detail needed</div>
        <div class="section-title">Your entry needs a bit more to work with</div>
    </div>
    """, unsafe_allow_html=True)
    for s in result.get("suggestions", []):
        st.markdown(f'<div class="growth-item">{s}</div>', unsafe_allow_html=True)


def _render_analysis_results(result: dict, mood, energy, stress):
    """Full analysis result display."""
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label" style="margin-bottom:12px;">Your analysis</div>
    """, unsafe_allow_html=True)

    # Top stats
    col1, col2, col3, col4 = st.columns(4)
    state = result.get("mental_state", {})
    sa    = result.get("self_awareness", {})
    ea    = result.get("emotional_analysis", {})
    mb    = result.get("mbti_primary", {})

    state_colors = {"Calm":"var(--green)","Reflective":"var(--accent2)","Analytical":"var(--accent)",
                    "Stressed":"var(--orange)","Overthinking":"var(--red)","Mixed":"var(--muted)"}
    sc = state_colors.get(state.get("label",""), "var(--muted)")

    with col1:
        st.markdown(f"""<div class="card"><div class="stat-label">Mental state</div>
        <div class="stat-value" style="color:{sc}; font-size:1.6rem;">{state.get('label','—')}</div></div>""",
        unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="card"><div class="stat-label">Self-awareness</div>
        <div class="stat-value" style="font-size:1.6rem;">{sa.get('score','—')}<span style="font-size:1rem;color:var(--muted)">/100</span></div></div>""",
        unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="card"><div class="stat-label">Emotional tone</div>
        <div class="stat-value" style="font-size:1.6rem;">{ea.get('tone','—')}</div></div>""",
        unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="card"><div class="stat-label">MBTI fit</div>
        <div class="stat-value" style="font-size:1.6rem;">{mb.get('type','—')}</div></div>""",
        unsafe_allow_html=True)

    # Tabs
    tab_ov, tab_ins, tab_per, tab_grow = st.tabs(["Overview", "Insights", "Personality", "Growth"])

    with tab_ov:
        render_overview_tab(result)
    with tab_ins:
        render_insights_tab(result)
    with tab_per:
        render_personality_tab(result)
    with tab_grow:
        render_growth_tab(result)

    st.markdown(f'<div class="disclaimer">{result.get("disclaimer","")}</div>', unsafe_allow_html=True)


# ============================================================================
# ANALYSIS TABS (preserved from original, no features removed)
# ============================================================================

def render_overview_tab(result: dict):
    ea = result.get("emotional_analysis", {})
    state = result.get("mental_state", {})

    if state.get("summary"):
        st.markdown(f'<div class="card-sm">{result.get("reflection_summary","")}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Emotional analysis**")
        for ins in ea.get("insights", [])[:3]:
            st.markdown(f'<div class="insight-item">{ins}</div>', unsafe_allow_html=True)
    with col2:
        sa = result.get("self_awareness", {})
        if sa:
            st.markdown("**Self-awareness breakdown**")
            bd = sa.get("breakdown", {})
            st.markdown(f'<div class="card-sm">'
                        f'<div class="stat-label">Clarity</div>{bd.get("clarity",0)}/100<br>'
                        f'<div class="stat-label" style="margin-top:8px;">Emotional stability</div>{bd.get("emotional_stability",0)}/100<br>'
                        f'<div class="stat-label" style="margin-top:8px;">Reflection depth</div>{bd.get("reflection_depth",0)}/100'
                        f'</div>', unsafe_allow_html=True)

    if result.get("highlighted_text_html"):
        st.markdown("**Language highlights**")
        st.markdown(f"""
        <div style="background:var(--surface2);border-radius:var(--radius-sm);padding:16px 18px;
             line-height:1.8;font-size:0.9rem;color:var(--text);border:1px solid var(--border);">
            {result['highlighted_text_html']}
        </div>""", unsafe_allow_html=True)
        for leg in result.get("highlight_legend", []):
            st.caption(leg)

    if result.get("timeline"):
        st.markdown("**Emotional arc**")
        timeline = result["timeline"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[t["step"] for t in timeline],
            y=[t["sentiment"] for t in timeline],
            mode="lines+markers",
            line=dict(color="#7c9ef8", width=2),
            marker=dict(size=6, color=[
                "#f07070" if t["label"] == "negative" else
                "#6edcaa" if t["label"] == "positive" else "#8a96b8"
                for t in timeline
            ]),
            hovertemplate="%{text}<extra></extra>",
            text=[t["sentence"][:60] + "…" for t in timeline],
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a96b8", size=12),
            xaxis=dict(title="Sentence", gridcolor="rgba(130,155,210,.08)"),
            yaxis=dict(title="Sentiment", gridcolor="rgba(130,155,210,.08)", range=[-1.1, 1.1]),
            margin=dict(l=0, r=0, t=10, b=0), height=220,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_insights_tab(result: dict):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Thought patterns**")
        for p in result.get("thought_patterns", []):
            st.markdown(f'<div class="insight-item">{p}</div>', unsafe_allow_html=True)
        st.markdown("**Communication style**")
        for s in result.get("communication_style", []):
            st.markdown(f'<div class="insight-item">{s}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Mental signals**")
        for s in result.get("mental_signals", []):
            st.markdown(f'<div class="insight-item">{s}</div>', unsafe_allow_html=True)

    if result.get("key_signals"):
        st.markdown("**Key writing signals**")
        for label, value, explanation in result["key_signals"]:
            pct = int(value * 100)
            st.markdown(f"""
            <div class="card-sm">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.9rem;font-weight:500;">{label}</span>
                    <span style="color:var(--accent);font-size:0.85rem;">{pct}%</span>
                </div>
                <p class="muted" style="font-size:0.82rem;margin:0;">{explanation}</p>
            </div>""", unsafe_allow_html=True)

    if result.get("confidence_note"):
        st.markdown(f'<div class="disclaimer">{result["confidence_note"]}</div>', unsafe_allow_html=True)


def render_personality_tab(result: dict):
    # Big Five radar
    latent = result.get("latent_profile", {})
    if latent:
        traits = list(latent.keys())
        scores = [latent[t]["score"] for t in traits]
        fig = go.Figure(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=traits + [traits[0]],
            fill="toself",
            fillcolor="rgba(124,158,248,0.15)",
            line=dict(color="#7c9ef8", width=2),
            marker=dict(size=6),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(130,155,210,.12)", tickfont=dict(color="#8a96b8", size=10)),
                angularaxis=dict(gridcolor="rgba(130,155,210,.12)", tickfont=dict(color="#e8edf8", size=12)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=30, b=30),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    # MBTI matches
    st.markdown("**MBTI personality fits**")
    for ins in result.get("mbti_insights", []):
        st.markdown(f'<div class="card-sm">{ins}</div>', unsafe_allow_html=True)

    # Questionnaire fusion info
    fusion = result.get("fusion", {})
    if fusion:
        tw = int(fusion.get("text_weight", 1.0) * 100)
        qw = int(fusion.get("questionnaire_weight", 0.0) * 100)
        st.markdown(f'<div class="disclaimer">Text analysis weighted at <strong>{tw}%</strong>, personality baseline at <strong>{qw}%</strong>.</div>',
                    unsafe_allow_html=True)


def render_growth_tab(result: dict):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your strengths in this entry**")
        for s in result.get("strengths", []):
            st.markdown(f'<div class="growth-item">{s}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Suggestions**")
        for s in result.get("suggestions", []):
            st.markdown(f'<div class="growth-item">{s}</div>', unsafe_allow_html=True)

    if result.get("reflection_prompts"):
        st.markdown("**Questions for your next entry**")
        for p in result["reflection_prompts"]:
            st.markdown(f'<div class="prompt-item">{p}</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE: TRENDS
# ============================================================================

def page_trends():
    entries = get_entries(st.session_state.username)

    st.markdown("""
    <div class="card">
        <div class="section-label">Your progress</div>
        <div class="display" style="font-size:1.8rem;">Mood &amp; mental state trends</div>
    </div>
    """, unsafe_allow_html=True)

    if len(entries) < 2:
        st.info("Write at least 2 entries to see trends here.")
        return

    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Mood timeline
    mood_order = {m: i for i, m in enumerate(DAILY_MOOD_OPTIONS)}
    df["mood_num"] = df["mood"].map(mood_order)

    fig = go.Figure()
    if "mood_num" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["mood_num"],
            mode="lines+markers",
            name="Mood",
            line=dict(color="#7c9ef8", width=2),
            marker=dict(size=8),
            hovertemplate="%{text}<extra></extra>",
            text=df["mood"].fillna("—"),
            yaxis="y1",
        ))
    if "self_awareness_score" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["self_awareness_score"],
            mode="lines+markers",
            name="Self-awareness",
            line=dict(color="#6edcaa", width=2, dash="dot"),
            marker=dict(size=6),
            yaxis="y2",
        ))
    if "stress" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["stress"],
            mode="lines",
            name="Stress",
            line=dict(color="#f07070", width=1.5),
            yaxis="y3",
        ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a96b8", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#e8edf8")),
        xaxis=dict(gridcolor="rgba(130,155,210,.08)"),
        yaxis=dict(title="Mood", gridcolor="rgba(130,155,210,.08)", tickvals=list(range(len(DAILY_MOOD_OPTIONS))), ticktext=DAILY_MOOD_OPTIONS),
        yaxis2=dict(title="Self-awareness", overlaying="y", side="right", range=[0, 100], showgrid=False),
        yaxis3=dict(visible=False, range=[0, 10]),
        height=320, margin=dict(l=0, r=60, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mental state distribution
    if "mental_state" in df.columns:
        st.markdown("**Mental state frequency**")
        counts = df["mental_state"].value_counts()
        state_colors_map = {"Calm": "#6edcaa","Reflective":"#b48bff","Analytical":"#7c9ef8",
                            "Stressed":"#f5a55a","Overthinking":"#f07070","Mixed":"#8a96b8"}
        fig2 = go.Figure(go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=[state_colors_map.get(s, "#8a96b8") for s in counts.index],
        ))
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a96b8"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(130,155,210,.08)"),
            height=220, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Energy vs stress scatter
    if "energy" in df.columns and "stress" in df.columns:
        st.markdown("**Energy vs stress**")
        fig3 = go.Figure(go.Scatter(
            x=df["energy"], y=df["stress"],
            mode="markers",
            marker=dict(size=10, color=df.get("mood_num", [5]*len(df)),
                        colorscale=[[0,"#7c9ef8"],[0.5,"#b48bff"],[1,"#6edcaa"]],
                        showscale=False),
            text=df["date"].dt.strftime("%b %d"),
            hovertemplate="%{text}<br>Energy %{x} · Stress %{y}<extra></extra>",
        ))
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a96b8"),
            xaxis=dict(title="Energy", gridcolor="rgba(130,155,210,.08)", range=[0, 11]),
            yaxis=dict(title="Stress",  gridcolor="rgba(130,155,210,.08)", range=[0, 11]),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ============================================================================
# PAGE: PERSONALITY
# ============================================================================

def page_personality():
    user_data = st.session_state.user_data
    name = user_data.get("name", "you")
    onboarding = user_data.get("onboarding_scores", {})

    st.markdown(f"""
    <div class="card">
        <div class="section-label">Personality overview</div>
        <div class="display" style="font-size:1.8rem;">{name}'s baseline profile</div>
        <p class="muted">Built from your onboarding answers and refined by your journal entries.</p>
    </div>
    """, unsafe_allow_html=True)

    if onboarding:
        dim_labels = {"IE": "Introvert ← → Extrovert", "NS": "Intuition ← → Sensing",
                      "TF": "Thinking ← → Feeling", "JP": "Judging ← → Perceiving"}
        for dim, score in onboarding.items():
            left_l, right_l = dim_labels.get(dim, dim).split(" ← → ")
            pct = int(score * 100)
            col1, col2, col3 = st.columns([2, 4, 2])
            with col1:
                st.markdown(f'<div style="text-align:right;color:var(--muted);font-size:0.85rem;padding-top:6px">{left_l}</div>', unsafe_allow_html=True)
            with col2:
                st.progress(score)
            with col3:
                st.markdown(f'<div style="color:var(--muted);font-size:0.85rem;padding-top:6px">{right_l} ({pct}%)</div>', unsafe_allow_html=True)

    entries = [e for e in get_entries(st.session_state.username) if e.get("mbti_primary")]
    if entries:
        from collections import Counter
        mbti_counts = Counter(e["mbti_primary"] for e in entries)
        top_type, top_count = mbti_counts.most_common(1)[0]
        st.markdown(f"""
        <div class="card" style="margin-top:16px;">
            <div class="section-label">Most frequent journal personality fit</div>
            <div class="stat-value" style="font-size:2.5rem; color:var(--accent2);">{top_type}</div>
            <p class="muted">Appeared in {top_count} of {len(entries)} analyzed entries.</p>
        </div>
        """, unsafe_allow_html=True)

        if len(mbti_counts) > 1:
            fig = go.Figure(go.Bar(
                x=list(mbti_counts.keys()),
                y=list(mbti_counts.values()),
                marker_color="#7c9ef8",
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8a96b8"),
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="rgba(130,155,210,.08)"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: HISTORY
# ============================================================================

def page_history(analyzer: PersonalityAnalyzer):
    entries = get_entries(st.session_state.username)

    st.markdown("""
    <div class="card">
        <div class="section-label">Past entries</div>
        <div class="display" style="font-size:1.8rem;">Entry history</div>
    </div>
    """, unsafe_allow_html=True)

    if not entries:
        st.info("No entries yet. Head to Today's Entry to write your first one.")
        return

    for entry in reversed(entries[-30:]):
        state_colors = {"Calm":"var(--green)","Reflective":"var(--accent2)","Analytical":"var(--accent)",
                        "Stressed":"var(--orange)","Overthinking":"var(--red)","Mixed":"var(--muted)","—":"var(--muted)"}
        sc = state_colors.get(entry.get("mental_state","—"), "var(--muted)")
        st.markdown(f"""
        <div class="card-sm" style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">
                <div>
                    <span style="font-weight:600;color:var(--text);">{entry['date']}</span>
                    &nbsp;<span style="color:var(--muted);font-size:0.82rem;">{entry.get('mood','')}</span>
                </div>
                <span style="color:{sc};font-size:0.82rem;font-weight:600;">{entry.get('mental_state','—')}</span>
            </div>
            <div style="color:var(--muted);font-size:0.85rem;line-height:1.5;">{entry.get('text_preview','')}</div>
            <div style="margin-top:8px;font-size:0.78rem;color:var(--muted);">
                {entry.get('word_count','—')} words &nbsp;·&nbsp;
                Self-awareness: {entry.get('self_awareness_score','—')} &nbsp;·&nbsp;
                MBTI: {entry.get('mbti_primary','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Compare two entries
    with st.expander("📊 Compare two entries"):
        st.markdown("Compare how your thinking and mood changed between two journal entries.")
        dates = sorted(set(e["date"] for e in entries), reverse=True)
        col1, col2 = st.columns(2)
        with col1:
            date1 = st.selectbox("Earlier entry date", dates, index=min(1, len(dates)-1), key="cmp_d1")
            text1 = st.text_area("Earlier entry text", height=120, key="cmp_t1")
        with col2:
            date2 = st.selectbox("Later entry date", dates, index=0, key="cmp_d2")
            text2 = st.text_area("Later entry text", height=120, key="cmp_t2")

        if st.button("⚖️ Compare", use_container_width=True):
            if not text1.strip() or not text2.strip():
                st.error("Please fill in both entries.")
            else:
                with st.spinner("Comparing…"):
                    comparison = analyzer.compare_texts(text1, text2)
                if not comparison.get("success"):
                    st.error(comparison.get("error", "Comparison failed."))
                elif comparison.get("safe_mode"):
                    st.warning("One entry triggered the safety layer. Comparison paused.")
                elif comparison.get("low_signal"):
                    st.info("One entry is too short for comparison.")
                else:
                    st.markdown(f"**{comparison.get('comparison_summary','')}**")
                    cols = st.columns(len(comparison.get("shift_cards", [])))
                    for col, card in zip(cols, comparison.get("shift_cards", [])):
                        with col:
                            st.markdown(f"""<div class="card-sm">
                            <div class="stat-label">{card['label']}</div>
                            <div style="color:var(--accent);font-weight:600;">{card['after']}</div>
                            <div class="muted">Was: {card['before']}</div>
                            </div>""", unsafe_allow_html=True)
                    if comparison.get("differences"):
                        df_diffs = pd.DataFrame(comparison["differences"])
                        st.dataframe(df_diffs, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Step 1: Auth gate
    if "username" not in st.session_state:
        render_auth()
        return

    # Step 2: Boot models with visible spinner (runs once, cached after)
    boot_models()

    # Step 3: Onboarding gate
    user_data = st.session_state.get("user_data", {})
    if not user_data.get("onboarding_done"):
        render_onboarding()
        return

    # Step 4: Load analyzer (already warm from boot_models)
    analyzer = load_analyzer()

    # Step 5: Sidebar nav
    page = render_sidebar()

    # Step 6: Route to page
    if page == "📝 Today's Entry":
        page_todays_entry(analyzer)
    elif page == "📊 My Trends":
        page_trends()
    elif page == "🧠 My Personality":
        page_personality()
    elif page == "📖 Entry History":
        page_history(analyzer)


if __name__ == "__main__":
    main()