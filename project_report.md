# 📊 Grammar Scoring Engine for Voice Samples — Project Report

## 🧠 Objective

The aim of this project is to **automatically evaluate spoken English grammar** using a combination of **traditional machine learning**, **deep learning (transformers)**, and **rule-based NLP** techniques. The system is designed to assign a **grammar score (0 to 5)** to each audio sample, reflecting the grammatical fluency of the speaker.

---

## 🚀 Pipeline Overview

The project follows a robust **multi-stage pipeline**, combining audio, text, NLP, and modelling layers:

### 1. **Audio Preprocessing**
- Raw `.wav` files are resampled to 16kHz.
- Silence and background noise are trimmed.
- Clean audio is saved for transcription.

### 2. **Speech-to-Text Transcription**
- Uses **Whisper (OpenAI)** for accurate transcription of spoken English.
- Output is a raw transcript for each audio file.

### 3. **Transcript Cleaning**
- Common disfluencies and fillers (e.g., “um”, “uh”, “you know”) are removed.
- Extra whitespaces and redundant punctuation are standardised.
- Text is lowercased for consistency.

### 4. **Feature Engineering**
- Extracts linguistic and grammar-related features:
  - `grammar_errors`: count via `language_tool_python`
  - `avg_sentence_length`: using spaCy sentence parser
  - `pos_diversity`: number of unique POS tags
  - `word_count`: total number of words
  - `grammar_errors_per_word`: normalised error rate

### 5. **Grammar Correction Estimation (GEC)**
- Uses T5-based model (`vennify/t5-base-grammar-correction`) to:
  - Generate corrected version of transcript
  - Compute word-level edits and edit ratios
  - Features: `gec_edits`, `gec_edit_rate`

### 6. **Model Training**
#### ✅ Traditional ML Models
- **Random Forest**, **Ridge**, and **LightGBM** are trained on feature data.
- Outputs are averaged as a traditional ensemble.

#### 🤖 Transformer-based Model
- **DistilBERT** is fine-tuned for regression using Hugging Face.
- Text-only input model capturing contextual grammar patterns.

#### 🧬 Meta-Ensemble Layer
- A **Linear Regression meta-model** is trained to combine:
  - DistilBERT prediction
  - Traditional ensemble prediction
- This produces the final grammar score.

---

## 🧪 Evaluation Metrics

| Metric         | Description                                |
|----------------|--------------------------------------------|
| **MAE**        | Mean Absolute Error – penalises large errors |
| **RMSE**       | Root Mean Square Error – penalises squared deviations |
| **Pearson r**  | Measures correlation with human labels     |

---

## 📊 Final Model Performance (on validation set)

| Metric         | Value    |
|----------------|----------|
| MAE            | 0.763 ✅ |
| RMSE           | 0.911 ✅ |
| Pearson Correlation | **0.625** ✅✅ |

This score reflects human-like ranking and prediction performance, despite a small training set (444 samples). The model is designed to scale, generalise, and integrate into real-world language assessment platforms.


---

## 📦 Output

- Final predictions are saved in `submission.csv` with columns:
  - `filename`: Audio filename
  - `label`: Predicted grammar score (integer, 0–5)

---

## 🔍 Tools & Libraries

| Component            | Libraries / Frameworks                          |
|---------------------|--------------------------------------------------|
| Audio Handling       | `librosa`, `soundfile`                          |
| ASR                  | `whisper` (OpenAI)                              |
| NLP & POS Parsing    | `spaCy`, `language_tool_python`                 |
| Grammar Correction   | `HappyTransformer` (T5 model)                   |
| Machine Learning     | `scikit-learn`, `lightgbm`, `numpy`, `pandas`   |
| Transformers         | `transformers`, `datasets`, `DistilBERT`        |
| Meta Ensemble        | `LinearRegression`                              |

---

## 📌 Limitations & Future Work

- **Accent Robustness**: Currently trained on existing transcripts; would benefit from multilingual and accented data fine-tuning.
- **Real-time Scoring**: Optimisation required for production deployment.
- **Explainability**: SHAP/LIME can be used for model interpretability.
- **Multimodal Learning**: Potential to fuse raw audio + text features in future iterations.

---

## 🧾 Conclusion

The grammar scoring engine provides a robust and scalable solution to automatically assess spoken English grammar. By fusing linguistic rules, machine learning, and deep NLP models, it achieves high correlation with human-annotated scores, making it suitable for use in assessments, interviews, and learning platforms.

---
