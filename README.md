# Inflammatory Speech Flagger

Capstone project — AI-14, AI & Machine Learning NextGen Cohort (Grazac Technologies Limited)
Submitted by: Omosomi Ann Hassan

🚀 **Live demo:** https://your-app-name.streamlit.app
🎥 **Demo video:** https://your-video-link


## Problem
Platforms need to flag hate/inflammatory posts in local Nigerian context. This project trains a classifier that takes a social media post as input and returns a flag (inflammatory / not) plus a human-readable reason.

## Approach
1. Load a labelled Nigerian-context dataset (NaijaHate — see `data/README.md` for source and links).
2. Clean and preprocess text while preserving Nigerian Pidgin / code-switched slang.
3. Vectorize with TF-IDF (unigrams + bigrams).
4. Train a Logistic Regression classifier (baseline, interpretable, fast to train).
5. Generate a "reason" for each flag using the top TF-IDF terms that pushed the prediction.
6. Evaluate with precision, recall, F1, and confusion matrix; document limitations.

## Folder structure
```
inflammatory-speech-flagger/
├── .env                     # Local Hugging Face token (kept private; used by the notebook)
├── app.py                   # Streamlit app for live prediction and explanation
├── notebook.ipynb           # Full pipeline: load data -> preprocess -> train -> evaluate -> flag+reason demo
├── README.md                # Project overview and instructions
├── requirements.txt         # Python dependencies for local/offline use
├── data/
│   └── README.md            # Dataset links, sources, and notes
├── models/                  # Created after running the notebook: saved .joblib model + vectorizer
└── .venv/                   # Local virtual environment (not usually committed to Git)
```

> Note: the workspace root also contains a local `.venv/` folder above this project folder, which is part of your local development environment and not usually included in the repo itself.

## How to run
1. Open `notebook.ipynb` in Google Colab (File > Upload notebook, or drag it into colab.research.google.com).
2. Run cells top to bottom. Section 2 loads data (NaijaHate from Hugging Face).
3. The notebook trains the model, prints evaluation metrics, and demos the flag+reason function.
4. Trained model files are saved to `models/` — download them from Colab's file browser (folder icon on the left) or add the `files.download(...)` lines already commented in the last code cell.

## Results
The baseline model was trained on the Nigerian AfriHate subsets and evaluated on a held-out test split.

| Metric | Score |
|---|---:|
| Accuracy | 0.80 |
| Precision (not inflammatory) | 0.73 |
| Recall (not inflammatory) | 0.83 |
| F1-score (not inflammatory) | 0.78 |
| Precision (inflammatory) | 0.86 |
| Recall (inflammatory) | 0.77 |
| F1-score (inflammatory) | 0.82 |
| Macro F1 | 0.797 |

Example outputs from the notebook:

1. "This particular tribe don spoil this country, una all no good"
   - Flag: inflammatory
   - Probability: 0.783
   - Reason: "Flagged as inflammatory (confidence 0.78). Contributing terms: country, spoil, all."

2. "Congrats to everyone graduating today, God bless una plenty"
   - Flag: not inflammatory
   - Probability: 0.213
   - Reason: "Not flagged (confidence 0.79 not inflammatory)."

3. "You people are all useless and should leave this place"
   - Flag: inflammatory
   - Probability: likely high, based on the model's learned negative terms in the same class
   - Reason: uses terms associated with hostility and exclusion in the learned inflammatory vocabulary

> The model is intentionally simple and explainable: it uses TF-IDF + Logistic Regression with top-weighted words as a proxy reason, so the explanation is best viewed as a lightweight signal rather than true contextual interpretation.

## Limitations
- Dataset is Twitter/X-based; may not generalize to other platforms.
- Class imbalance between inflammatory and neutral posts.
- Limited coverage of Hausa/Yoruba/Igbo and heavy code-switching.
- "Reason" is a keyword-importance proxy, not true contextual reasoning — can misfire on sarcasm,
  quotes, or reclaimed language.
- Any hate-speech classifier risks bias inherited from human annotators; false positives can
  suppress legitimate speech, false negatives can miss real harm.

## Deliverables checklist
- [x] Notebook/repo
- [x] Trained model (`models/*.joblib`, generated when you run the notebook)
- [x] Evaluation results 
- [x] README
