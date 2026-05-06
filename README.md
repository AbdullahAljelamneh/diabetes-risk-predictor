# 🩺 Diabetes Risk Predictor

An end-to-end machine learning system that predicts diabetes risk from clinical variables, with full model explainability and a live web interface.

**[🚀 Live Demo](https://diabetes-risk-predictor-stz43dnvcjgahn3icjc2vk.streamlit.app/)**

---

## Project Overview

This project is part of a healthcare AI learning roadmap (Phase 6 Capstone).
It covers the full ML pipeline from raw data to a deployed web application.

---

## Pipeline

- **Data:** Pima Indians Diabetes Dataset (768 patients, 8 clinical features)
- **Preprocessing:** KNN imputation for biologically impossible zeros
- **Feature Engineering:** 3 new clinical interaction features (Glucose/Insulin ratio, BMI×Age, Glucose×BMI)
- **Models trained:** Logistic Regression, Random Forest, SVM, XGBoost
- **Tuning:** Optuna hyperparameter optimization (50 trials per model)
- **Explainability:** SHAP values with summary plot
- **Interface:** Streamlit web app deployed on Streamlit Cloud

---

## Results

| Metric | Score |
|--------|-------|
| CV Accuracy | 78.8% |
| Test Accuracy | 69% |
| ROC-AUC | 0.81 |

**Top predictive features (SHAP):** Glucose, BMI, Pregnancies, Insulin, Glucose/Insulin Ratio

---

## Tech Stack

Python, Scikit-learn, XGBoost, Optuna, SHAP, Streamlit, Pandas, NumPy

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

> ⚠️ This tool is for educational purposes only and is not a medical diagnosis.
