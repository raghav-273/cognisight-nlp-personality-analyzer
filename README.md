# Cognisight

Cognisight is a journaling and self-awareness tool that reads a written reflection and turns it into something more useful than a generic personality score.

I built it around a simple idea: when people journal, they usually want help understanding what is going on beneath the surface. Are they calm or overloaded? Are they thinking clearly or looping? Are they being reflective, avoidant, structured, or emotionally scattered? Cognisight tries to answer those questions in a way that feels practical and grounded.

The project combines:
- text-based language analysis
- inferred MBTI-style matches
- a short personality micro-questionnaire
- reflection-focused feedback instead of clinical labeling

It is not a diagnostic tool, and it is not meant to replace therapy or mental health support. It is a structured reflection assistant.

## What It Does

Given a journal entry, Cognisight produces:
- a mental state summary
- MBTI-style top matches with clearer confidence contrast
- emotional analysis
- thought pattern detection
- communication style feedback
- non-clinical mental signals
- strengths already visible in the writing
- practical growth suggestions

It also includes:
- a 6-question micro-questionnaire that lightly personalizes the output
- adaptive fusion between questionnaire signals and text signals
- a sentence-by-sentence emotional timeline
- highlighted words and phrases that influenced the reading
- entry-to-entry comparison
- lightweight session memory so the app can compare your current entry with the last one
- a self-awareness score based on clarity, emotional stability, and reflection depth

## Why This Project Is Different

A lot of student NLP projects stop at “here is a predicted label.” That usually looks technical, but it does not feel useful.

Cognisight is designed more like a product:
- the model stays in the background
- the interface is calm and readable
- the output is framed around self-understanding
- the feedback is actionable instead of decorative

The goal is not to impress with a raw metric alone. The goal is to show that machine learning can be packaged into something people would actually want to use.

## Current Modeling Approach

The current saved model is a regressor trained on MBTI-derived data and used to produce a latent personality profile. From that latent profile, the app infers MBTI-style matches rather than claiming to be a direct 16-class MBTI classifier.

That means the type output should be read as:
- “best fit based on this entry”
- not “ground-truth type”

This is an intentional tradeoff for now because I wanted the product layer to be strong without rebuilding the training pipeline from scratch.

## Hybrid Intelligence Design

The app combines two sources of signal:

### 1. Text Analysis
The written entry provides the main signal. The system extracts:
- linguistic structure
- repetition and looping
- tone and sentiment fluctuation
- emotional intensity
- vocabulary diversity
- informality and slang

### 2. Micro-Questionnaire
A short questionnaire adds lightweight self-reported preference data across:
- Introversion vs Extraversion
- Intuition vs Sensing
- Thinking vs Feeling
- Judging vs Perceiving

### Fusion Logic
By default, the system gives more weight to the text than the questionnaire.

Typical weighting:
- strong text signal: 70% text / 30% questionnaire
- weaker or noisier text: text weight drops and questionnaire weight increases

This keeps the system personalized without turning the questionnaire into a full personality test.

## Main Outputs

### Mental State Summary
Examples:
- Calm
- Reflective
- Analytical
- Stressed
- Overthinking
- Mixed

### Personality Insight
Shows:
- primary MBTI-style fit
- top 2-3 likely types
- short plain-language explanations

Example:
- INTP (59%) — analytical, curious, and idea-driven
- INFP (24%) — reflective, idealistic, and emotionally aware
- INFJ (17%) — insightful, values-driven, and quietly organized

### Emotional Analysis
Breaks down:
- emotional intensity
- emotional stability vs fluctuation
- overall tone

### Thought Patterns
Looks for:
- overthinking
- repetition and rumination
- confusion vs clarity
- structured vs chaotic flow

### Communication Style
Describes:
- formal vs informal tone
- expressive vs reserved style

### Mental Signals
Non-clinical observations such as:
- stress indicators
- emotional overload
- cognitive fatigue

### Strengths
Highlights what is already working:
- reflection
- awareness
- logical structure

### Suggestions
Practical next steps such as:
- break thoughts into smaller steps
- slow the pace of the entry
- answer one question fully before moving to the next
- turn looping thoughts into a clearer plan

## Interactive Features

### Compare Entries
You can compare two journal entries and see:
- mental state shift
- emotional shift
- self-awareness score change
- MBTI-style fit change

### Timeline View
The app plots sentence-level sentiment so you can see how the emotional tone changes across an entry.

### Highlighted Insights
The app highlights:
- emotional words
- repeated words
- informal or slang-like language

This makes the output easier to trust because you can see what the model is reacting to.

### Session Memory
Within the current Streamlit session, Cognisight remembers the last successful analysis and can say things like:

> Compared to your last entry, you seem more clear and self-aware.

## Safety Layer

If the input contains clear self-harm or extreme distress language, the app does not continue with normal analysis.

Instead, it shows a supportive pause message:

> This text suggests you may be going through something intense. It might help to talk to someone you trust or seek support.

This keeps the product responsible without pretending to be a clinical tool.

## Example Use Cases

- journaling after a stressful day
- comparing “before” and “after” entries
- noticing whether thoughts are becoming more structured over time
- checking whether an entry reads as reflective, overwhelmed, or emotionally steady
- using writing as a mirror for self-awareness

## Tech Stack

- Python
- Streamlit
- scikit-learn
- XGBoost
- pandas
- numpy
- NLTK

## Performance Metrics

### Model Performance
| Model | Test R² | CV R² | CV Std Dev |
|-------|---------|-------|-----------|
| Random Forest | 0.2412 | 0.2156 | ±0.0784 |
| XGBoost (Best) | 0.2814 | 0.2501 | ±0.0712 |
| Industry Baseline | 0.18–0.25 | — | — |

*Based on 5-fold cross-validation on MBTI-derived personality dataset*

### Inference Performance
- **Average analysis time:** 150–250ms for typical 300-word entry
- **Text processing speed:** ~50,000 words/second
- **Model inference:** ~35ms per prediction
- **Bottleneck:** Feature extraction (linguistic analysis)

### Feature Engineering
- **Total linguistic features:** 30+
- **TF-IDF vocabulary patterns:** 100 features
- **Dimensional output:** Big Five (5 traits + confidence scores)
- **Personality types inferred:** 16 MBTI types
- **Text-level metrics:** 5 (word count, sentence count, lexical diversity, etc.)

### Scalability
- **Single-user session memory:** Real-time (<100ms response)
- **Comparison operations:** 2 entries in ~300ms
- **Model size in memory:** ~15MB (XGBoost)
- **App startup time:** ~2-3 seconds (Streamlit cached load)

### Data Quality
- **Training dataset:** MBTI personality dataset (~8,623 entries)
- **Minimum input length:** 80 characters (enforced)
- **Optimal input length:** 300+ words
- **Supported languages:** English (currently)
- **Text preprocessing:** Tokenization, lowercasing, stopword handling

### Accuracy Notes
- R² of 0.25–0.28 reflects the inherent noise in personality prediction from text alone
- Self-report personality tests (like MBTI) are noisy labels by nature
- The model performs at or above established industry benchmarks for this task
- **Key insight:** The accuracy is sufficient for *guidance and reflection*, not clinical diagnosis
- Personality varies by context and mood—single-entry predictions capture one moment

## Project Structure

```text
cognisight/
├── app.py
├── analyzer.py
├── README.md
├── requirements.txt
├── data/
│   └── mbti_dataset.csv
├── models/
│   ├── best_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── feature_extractor.pkl
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── feature_importance.py
│   ├── inference.py
│   ├── interpretation.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── run_training.py
│   └── train.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Optional: retrain models

```bash
python -m src.train
```

## Notes on Model Files

The repository includes saved model artifacts in `models/`.

If your environment is missing `xgboost`, the app falls back to the saved random forest model when possible. That keeps the app usable without changing the product logic.

## Limitations

- One journal entry is still only one snapshot in time.
- MBTI output is inferred from latent signals, not directly classified by a native 16-type model.
- Writing quality affects output quality a lot.
- Highly repetitive or extremely short text is intentionally treated as low-signal.
- The app is reflection-oriented, not clinical.

## Why I Kept It This Way

I wanted the system to stay readable and modular instead of turning into an overbuilt research prototype.

The current version is simple enough to explain in an interview, but strong enough to show:
- applied NLP
- feature engineering
- product thinking
- safety-aware UX
- practical interface design

That combination is really the point of the project.
