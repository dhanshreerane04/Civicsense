# CivicSense 🇮🇳

### Multilingual Citizen Grievance Classification System

CivicSense is a multilingual citizen grievance classification system designed to automatically identify the category of citizen complaints written in **Hindi, Hinglish, and English**.

The system uses a fine-tuned **IndicBERT** model along with a keyword-assisted hybrid prediction layer to classify grievances into 14 civic service categories.

---

## 🎯 Problem Statement

Citizens submit complaints about public services in different languages and writing styles. Manually sorting these grievances into departments can be time-consuming.

CivicSense aims to automatically classify a citizen's complaint into the appropriate service category from its text.

---

## ✨ Features

- Multilingual grievance classification
- Supports Hindi, Hinglish, and English text
- 14 grievance categories
- Fine-tuned IndicBERT model
- Keyword-assisted hybrid prediction
- Confidence-based prediction logic
- Template-aware dataset evaluation
- Separate unseen evaluation dataset
- Streamlit-based interface

---

## 🗂️ Grievance Categories

CivicSense supports the following 14 categories:

1. Banking & Financial Services
2. Corruption & Bribery
3. Education & Schools
4. Electricity
5. Employment & Labour
6. Healthcare & Hospitals
7. Land Records & Revenue
8. Municipal Certificates
9. Pension & Provident Fund
10. Police & Law and Order
11. Ration & Public Distribution System
12. Roads & Infrastructure
13. Sanitation & Garbage
14. Water Supply

---

## 📊 Dataset

The project uses a multilingual citizen grievance dataset containing **614 labelled complaints**.

Each record contains:

- `grievance_id`
- `text`
- `category`

The dataset contains complaints written using Hindi, Hinglish/Roman text, and English.

A separate **98-sample unseen evaluation dataset** was also created to evaluate how well the model handles complaints that were not part of the original dataset.

---

## 🔍 Data Analysis

Initial data analysis included:

- Missing-value analysis
- Text cleaning
- Text-length analysis
- Category distribution
- TF-IDF feature analysis
- Cosine similarity analysis
- Detection of repeated/template-like complaints
- Template-aware grouping
- Group-based train/test splitting

The dataset contained repeated template-like complaint structures. Therefore, similarity-based groups were created before performing a group-aware evaluation.

---

## 🧪 Baseline Model

A TF-IDF based text classification approach was implemented as a baseline.

The following traditional machine-learning models were evaluated:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

The TF-IDF + Logistic Regression baseline achieved 100% accuracy on the template-aware held-out evaluation.

Because the dataset contains structured/template-like complaints, this result was not treated as evidence of real-world 100% accuracy.

---

## 🤖 IndicBERT

For multilingual text classification, the project uses:

**Model:** `ai4bharat/IndicBERTv2-MLM-only`

The pretrained IndicBERT model was fine-tuned for the 14-class grievance classification task.

### Training Configuration

- Maximum sequence length: 128
- Epochs: 3
- Batch size: 8
- Learning rate: 2e-5
- Weight decay: 0.01
- GPU acceleration used during training

---

## 🔀 Hybrid Prediction Approach

The final prediction system combines:

**IndicBERT + keyword-assisted classification logic**

The system first obtains IndicBERT predictions and confidence scores.

Domain-specific keywords are then used to assist the prediction when:

- multiple strong category keywords are detected, or
- IndicBERT confidence is low and a relevant keyword is detected.

This helps the system handle domain-specific civic terminology and ambiguous complaints.

---

## 📈 Evaluation

The final system was evaluated on a separately curated **98-sample unseen evaluation set** covering all 14 categories.

### Results

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 100.00% | 100.00% | 100.00% | 1.0000 |
| IndicBERT | 70.41% | 71.63% | 70.41% | 0.6896 |
| **IndicBERT + Keyword Hybrid** | **94.90%** | **95.61%** | **94.90%** | **0.9484** |

### Hybrid Improvement

On the unseen evaluation set:

- Accuracy improved from **70.41% → 94.90%**
- Improvement: **+24.49 percentage points**
- Macro F1 improved from **0.6896 → 0.9484**

The hybrid approach also improved classification of the Roads & Infrastructure category, where the standalone IndicBERT model struggled on the unseen examples.

> The 94.90% result is based on a separately curated 98-sample evaluation set and should not be interpreted as production or real-world accuracy.

---

## 🏗️ Project Architecture

```text
                 Citizen Complaint
                         │
                         ▼
                  Text Preprocessing
                         │
                         ▼
                   IndicBERT Model
                         │
                         ▼
                 Prediction + Confidence
                         │
                         ▼
              Keyword-Assisted Layer
                         │
                         ▼
                Final Category Output