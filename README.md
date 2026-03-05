# SHL Audio Grammar Scoring Challenge

Predicting grammar scores (0–5) from spoken audio samples using Whisper transcription, linguistic + acoustic feature extraction, and ensemble regression.

## Approach

### 1. Audio Transcription
All audio files (45–60s each) are transcribed to text using OpenAI's **Whisper** (base model).

### 2. Feature Extraction

| Feature Group | Count | Description |
|---|---|---|
| Linguistic | 23 | Word count, vocabulary richness, POS distributions, lexical density, syllable complexity, type-token ratio |
| TF-IDF (Word) | 300 | Word-level n-grams (1–2) from transcripts |
| TF-IDF (Char) | 200 | Character-level n-grams (3–5) from transcripts |
| Acoustic | ~100 | MFCCs + deltas, spectral features, ZCR, RMS energy, chroma, tempo, silence/pause metrics |

### 3. Modeling
Multiple regressors trained with 5-fold cross-validation:
- Ridge, ElasticNet, SVR (RBF + Linear)
- Random Forest, Gradient Boosting
- XGBoost, LightGBM

Top models combined via **Stacking** (Ridge meta-learner) and **Weighted Blending** (Pearson-based weights). Best approach selected automatically.

### 4. Evaluation
- Primary metric: **Pearson Correlation**
- Secondary: RMSE, MAE, R²

## Dataset
- **Training**: 409 audio samples with MOS Likert Grammar Scores (0–5)
- **Testing**: 197 audio samples

## Requirements
```
openai-whisper
librosa
scikit-learn
xgboost
lightgbm
nltk
numpy
pandas
scipy
```

## Usage
Run `shl-audio-scoring-solution.ipynb` on Kaggle with the competition dataset attached, or locally with the dataset in the expected directory structure.
