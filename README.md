# Complaint AI System

## Overview
This upgraded MSc-level project converts free-text complaints into structured, explainable complaint intelligence with three competing pipelines:
- Rule-Based reasoning
- Classical ML classifier
- Retrieval-Augmented complaint analysis with FAISS similarity search

The system outputs:
- `core_issue`
- `detected_entities`
- `department`
- `priority`
- `actionable_task`
- `confidence_score`
- `reasoning`

## Data Setup
- `data/raw/complaints.csv`: 50,000-row synthetic complaint warehouse used for training, retrieval, and large-scale analytics
- `data/processed/cleaned_complaints.csv`: processed 50,000-row version with `cleaned_text`
- `data/validation/handwritten_complaints.csv`: 5,000-row handwritten-style validation set for more realistic language variation
- `data/cfpb/consumer_complaints.csv`: separate CFPB-style benchmark dataset mapped to the project's four departments
- `models/ml_classifier.pkl`: persisted ML classifier trained from the large dataset
- `models/vector_index.faiss`: FAISS index built from the same complaint warehouse

This keeps raw data, processed data, model artifacts, analytics, and evaluation on the same project scale.

Raw vs handwritten:
- They use the same schema so the same model code and dashboard can evaluate both datasets.
- `raw` is the large synthetic warehouse for training and retrieval.
- `handwritten` is a separate realism-check dataset with more natural wording and less rigid templating.

## Key Features
- Multi-step complaint analysis with JSON transparency
- FAISS similarity search over complaint embeddings
- Sentence-transformer embeddings with CPU-safe fallback when model weights are unavailable offline
- Complaint clustering with TF-IDF + KMeans
- Analytics charts using matplotlib and seaborn
- Multi-page Streamlit dashboard with lazy-loaded heavy pages
- Model comparison using grouped holdout evaluation, accuracy, precision, recall, and F1 score
- Gemini on/off toggle with vision support for multimodal LLM analysis
- Text + image complaint analysis on the analyzer page
- 50,000-row complaint warehouse for major-project experimentation
- Handwritten validation set for stronger academic credibility
- CFPB-style external benchmark page
- Smoke-test suite for schema, rule-based, ML, and RAG execution

## Run
```powershell
cd C:\Users\Admin\Documents\Playground\complaint_ai_system
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

## Refresh Processed Data
```powershell
.\venv\Scripts\python.exe -m src.preprocessing
```

## Evaluation Protocol
- The benchmark now uses grouped holdout splitting instead of a plain random split.
- Complaint templates are normalized into evaluation groups before splitting, so near-duplicate synthetic complaints do not appear in both train and test.
- This makes the reported scores more defensible in a major-project report or viva.
- The comparison page also reports results on a separate handwritten validation set so you can discuss generalization beyond the generated warehouse.
- The CFPB benchmark page shows transfer performance on a separate external-style dataset.

## Run Smoke Tests
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Validation Dataset
- Use `data/validation/handwritten_complaints.csv` as a small realism check outside the synthetic warehouse.
- This dataset is intentionally written by hand and should be discussed in the project report as an external validation slice.

## Gemini Toggle
- The analyzer page includes an on/off toggle named `Use Gemini vision`.
- When enabled, the RAG + LLM panel uses the Gemini API key configured through environment variables or Streamlit secrets and can process complaint text together with an uploaded image.
- When disabled, the project stays in deterministic/offline analysis mode.

## Streamlit Community Cloud
- Recommended entrypoint: `app/streamlit_app.py`
- Add `GEMINI_API_KEY` or `EMBEDDED_GEMINI_API_KEY` in your Streamlit app secrets before enabling Gemini vision.
- Community Cloud deployment steps are documented here: [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Notes
- If the environment blocks Hugging Face downloads, the embedding layer automatically falls back to a local hashing embedding backend while still using FAISS.
- The dashboard now avoids remote font downloads so that non-LLM functionality stays offline-friendly.
- The current project is aligned to a large-dataset MSc project structure, but report quality still depends on how honestly you present model limitations and evaluation scope.
