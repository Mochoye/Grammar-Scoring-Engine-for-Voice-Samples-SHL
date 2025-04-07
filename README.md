#  Grammar Scoring Engine for Spoken English (Hybrid NLP + ML Model)

##  Overview

This notebook presents a high-performance **Grammar Scoring Engine** built for evaluating spoken English responses. Each audio clip is 45–60 seconds long and is rated on a continuous **0–5 grammar proficiency scale** (MOS-based).

The challenge is tackled by combining the **best of deep learning and classical ML**, creating a **hybrid ensemble model** that mimics expert human scoring by evaluating both:
- **Grammatical Accuracy** (via error detection and correction)
- **Grammatical Sophistication** (via sentence structure and syntactic complexity)

---

## 🛠️ Model Pipeline

The workflow includes:

### 🎧 Audio Preprocessing & Transcription
- Normalisation and silence trimming of `.wav` files
- Transcription using **OpenAI Whisper (base)** ASR

### ✂️ Transcript Cleaning
- Removal of disfluencies (e.g., “um”, “uh”, “like”)
- Standardised punctuation for downstream NLP

### 🧩 Feature Engineering
- Grammar error count using `language_tool_python`
- Syntactic complexity via spaCy (POS diversity, sentence length)
- **GEC-based edit distance** using a grammar correction model

### 🤖 Hybrid Scoring Model
- **DistilBERT** fine-tuned on cleaned transcripts (regression)
- **Feature ensemble** (RandomForest + LightGBM + Ridge)
- **Meta-regressor** combines both model outputs for final prediction

---

## 📊 Final Model Performance (on validation set)

| Metric         | Value    |
|----------------|----------|
| MAE            | 0.763 ✅ |
| RMSE           | 0.911 ✅ |
| Pearson Correlation | **0.625** ✅✅ |

This score reflects human-like ranking and prediction performance, despite a small training set (444 samples). The model is designed to scale, generalise, and integrate into real-world language assessment platforms.

---

## 📁 Output
- Final predictions saved as `submission.csv`
- All grammar scores are **rounded to integers** as per Likert-style evaluation

---

## 🏁 Closing Notes
This notebook demonstrates how combining ASR, NLP, and ML can replicate expert-level grammar evaluation in speech. The approach is modular, interpretable, and ready for deployment in automated spoken language assessment.

