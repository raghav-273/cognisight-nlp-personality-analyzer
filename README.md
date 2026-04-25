<div align="center">

<br/>

```
   ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗
  ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝
  ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║███████╗██║██║  ███╗███████║   ██║   
  ██║     ██║   ██║██║   ██║██║╚██╗██║██║╚════██║██║██║   ██║██╔══██║   ██║   
  ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║███████║██║╚██████╔╝██║  ██║   ██║   
   ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝  
```

**A journaling intelligence tool that turns your writing into self-understanding.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22C55E?style=flat-square)]()

<br/>

> *Most journaling apps give you a word cloud. Cognisight gives you a mirror.*

<br/>

</div>

---

## What Is Cognisight?

Cognisight reads a written journal entry and turns it into something more useful than a generic personality score.

When people journal, they want help understanding what's going on beneath the surface — **Are they calm or overloaded? Thinking clearly or looping? Being reflective, avoidant, structured, or emotionally scattered?** Cognisight answers those questions in a way that feels practical and grounded, not clinical.

It is **not a diagnostic tool**. It is a structured reflection assistant — closer to a thoughtful reading of your writing than a psychological assessment.

<br/>

## Features at a Glance

<table>
<tr>
<td width="50%" valign="top">

### Core Analysis
- **Mental state classification** — Calm, Reflective, Analytical, Stressed, Overthinking, Mixed
- **Big Five personality inference** — from latent writing signals
- **MBTI-style fit** — top 2–3 types with confidence and plain-English explanation
- **Self-awareness score** — clarity + emotional stability + reflection depth

</td>
<td width="50%" valign="top">

### Language Insights
- **Thought pattern detection** — rumination, looping, clarity vs chaos
- **Emotional timeline** — sentence-by-sentence sentiment arc
- **Highlighted text** — shows exactly what influenced the reading
- **Communication style** — formal vs informal, expressive vs reserved

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Tracking & Memory
- **User accounts** — private entries stored per user
- **Daily check-ins** — mood, energy, stress sliders + reflection prompt
- **Trend charts** — mood, mental state, self-awareness over time
- **Entry comparison** — mental state shift, emotional shift, type change

</td>
<td width="50%" valign="top">

### Responsible Design
- **Safety layer** — detects extreme distress language and pauses analysis
- **Non-clinical framing** — mental signals, not diagnoses
- **Adaptive fusion** — balances text signal vs questionnaire signal automatically
- **Onboarding baseline** — personality context from first use

</td>
</tr>
</table>

<br/>

## How It Works

```
               Your journal entry
                       │
                       ▼
  ┌─────────────────────────────────────────────────┐
  │              TEXT ANALYSIS                      │
  │  • 30 linguistic features                       │
  │    (tone, structure, repetition, variance...)   │
  │  • 100 TF-IDF vocabulary patterns               │
  │  • 7-class emotion model (HuggingFace)          │
  │  • Sentence-level sentiment arc                 │
  └────────────────────┬────────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────────┐
  │           PERSONALITY BASELINE                  │
  │  • 20-question onboarding (Big Five + context)  │
  │  • Adaptive fusion: strong text = 70% text      │
  │                     weak text  = 50/50 split    │
  └────────────────────┬────────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────────┐
  │                 ML MODEL                        │
  │  XGBoost regressor → Big Five latent profile    │
  │  → MBTI-style type inference (16 types)         │
  │  → Mental state classification                  │
  │  → Self-awareness score                         │
  └─────────────────────────────────────────────────┘
                       │
                       ▼
       Insights  ·  Strengths  ·  Suggestions
       Emotional timeline  ·  Reflection prompts
```

<br/>

## Performance

<table>
<tr>
<th align="left">Model</th>
<th align="center">Test R²</th>
<th align="center">CV R² (5-fold)</th>
<th align="center">CV Std Dev</th>
</tr>
<tr>
<td>Random Forest</td>
<td align="center">0.2412</td>
<td align="center">0.2156</td>
<td align="center">±0.0784</td>
</tr>
<tr>
<td><strong>XGBoost (Best)</strong></td>
<td align="center"><strong>0.2814</strong></td>
<td align="center"><strong>0.2501</strong></td>
<td align="center">±0.0712</td>
</tr>
<tr>
<td>Industry Baseline</td>
<td align="center">0.18–0.25</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
</table>

> **On R² for personality prediction:** R² ≈ 0.25–0.28 is realistic and at or above published benchmarks for this task. Personality labels derived from MBTI are inherently noisy self-reports. The model is accurate enough for *reflection and guidance* — which is exactly what it's built for.

### Inference Speed

| Operation | Time |
|---|---|
| Full analysis (300-word entry) | 150–250ms |
| Model inference only | ~35ms |
| Emotion model (transformer) | ~2–3s (CPU, cached after first run) |
| Entry comparison | ~300ms |
| App startup | ~2–3s (cached after first load) |

<br/>

## Project Structure

```
cognisight/
├── app.py                    ← Streamlit UI, user auth, daily check-in flow
├── analyzer.py               ← Main analysis orchestrator
│
├── src/
│   ├── feature_extractor.py  ← 30 linguistic + 100 TF-IDF features
│   ├── emotion_extractor.py  ← HuggingFace transformer (7-class emotion + sentiment)
│   ├── interpretation.py     ← Mental state, MBTI inference, insights, strengths
│   ├── inference.py          ← Model loading and prediction pipeline
│   ├── feature_importance.py ← Feature contribution and explainability
│   ├── preprocessing.py      ← Tokenisation, cleaning, POS tagging
│   ├── model.py              ← Model factory and manager
│   ├── train.py              ← Training pipeline with GridSearchCV + CV
│   └── run_training.py       ← Training entry point
│
├── utils/
│   ├── helpers.py            ← Config, MBTI→Big Five mapping, normalization
│   └── __init__.py
│
├── models/
│   ├── best_model.pkl        ← XGBoost (loaded by default)
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── feature_extractor.pkl ← Fitted TF-IDF vectorizer
│
└── data/
    └── mbti_dataset.csv      ← 8,623 MBTI-labelled writing samples
```

<br/>

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/yourusername/cognisight.git
cd cognisight
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

On first run, the emotion analysis models (~500MB) download automatically from HuggingFace and are cached locally. This is a one-time cost — every subsequent run loads from cache in ~2 seconds.

### 3. Retrain models (optional)

```bash
python -m src.train
```

This runs full GridSearchCV hyperparameter tuning and 5-fold cross-validation on the MBTI dataset and saves new model artifacts to `models/`.

<br/>

## Dependencies

```
streamlit          # UI framework
scikit-learn       # Random Forest, preprocessing
xgboost            # Best-performing model
transformers       # HuggingFace emotion + sentiment pipelines
torch              # Transformer backend
nltk               # Tokenisation, POS tagging, VADER sentiment
pandas             # Data handling
numpy              # Feature vectors
plotly             # Charts and visualisations
```

Install everything at once:

```bash
pip install -r requirements.txt
```

> **Note on XGBoost:** If `xgboost` is unavailable, the app automatically falls back to the saved Random Forest model. No code changes needed.

<br/>

## Design Philosophy

Most student NLP projects stop at "here is a predicted label." That looks technical, but it doesn't feel useful.

Cognisight is designed more like a product:

- **The model stays in the background.** Users see insights, not feature vectors.
- **The output is framed around self-understanding**, not scoring or ranking.
- **Feedback is actionable** — concrete next steps, not decorative labels.
- **Safety is built in** — distress detection pauses analysis and shows support resources.
- **The interface is calm.** Dark, readable, and free of noise.

The goal isn't to impress with a raw metric. It's to show that machine learning can be packaged into something people would actually want to use.

<br/>

## Honest Limitations

- One journal entry is a snapshot in time. Personality varies by context and mood.
- MBTI output is inferred from latent signals — read it as "best fit for this entry," not ground truth.
- Writing quality matters. Short, repetitive, or extremely vague entries produce lower-signal output.
- English only (currently).
- Not a clinical tool. Not a replacement for therapy or professional mental health support.

<br/>

## Example Output

Given a journal entry like:

> *"I've been going over the same problem for hours and I can't seem to get out of my own head. I know what I should do, but every time I start, something pulls me back. I'm worried I'm running out of time and I don't know how to slow down..."*

Cognisight might return:

```
Mental state     →  Overthinking
Self-awareness   →  72 / 100
Emotional tone   →  Anxious-reflective
MBTI fit         →  INFP (61%)  ·  INFJ (22%)  ·  INTP (17%)

Thought patterns →  Rumination loop detected
                    Clarity visible in final sentence
                    High self-awareness despite overload

Strengths        →  Genuine reflection present
                    Clear awareness of the pattern

Suggestions      →  Write the one thing you'd do if you had to pick just one
                    Separate what you can control from what you can't
```

<br/>

## Safety Layer

If an entry contains clear self-harm or extreme distress language, the app does not continue with normal analysis. It shows a calm supportive message and crisis resources:

```
This text suggests you may be going through something intense.
It might help to talk to someone you trust or seek support.

iCall India:              9152987821
Vandrevala Foundation:    1860-2662-345  (24/7)
```

<br/>

---

<div align="center">

Built by Raghav Mishra with intention and care — a reminder that machine learning can be both intelligent and human-centered.

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>
