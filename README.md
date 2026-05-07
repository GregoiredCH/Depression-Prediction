# Depression Prediction

A machine learning project that predicts whether someone may be experiencing depression based on their responses to a short questionnaire. Built using data from the **NHANES survey (2005–2018)**, the project covers the full ML pipeline: data exploration, feature selection, model training, and a web-based interface for predictions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Models](#models)
- [Web Interface](#web-interface)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

---

## Overview

This project uses supervised machine learning to classify individuals as potentially depressed or not, based on questionnaire answers. A Flask web app allows users to fill out the questionnaire and receive a prediction instantly, powered by a pre-trained model loaded via pickle.

---

## Dataset

**Source:** [NHANES (National Health and Nutrition Examination Survey)](https://www.cdc.gov/nchs/nhanes/index.htm)  
**Period covered:** 2005 – 2018  
**Type:** Self-reported health and lifestyle questionnaire data

---

## ML Pipeline

### 1. Data Exploration
Exploratory data analysis was conducted using **pandas**, **matplotlib**, and **seaborn** to understand distributions, class balance, and relationships between features.

### 2. Feature Selection
Feature selection was done in three steps:
- **Correlation matrix** — to identify and remove highly correlated/redundant features
- **Feature importance** — using a Random Forest to rank features by predictive power
- **PCA (Principal Component Analysis)** — to reduce dimensionality and remove noise

### 3. Model Training
Multiple classifiers were trained and compared using **scikit-learn**.

### 4. Model Persistence
The best-performing model was serialized using **pickle** and is loaded at runtime by the Flask app.

---

## Models

The following classifiers were trained and evaluated:

| Model | Library |
|---|---|
| K-Nearest Neighbors (KNN) | scikit-learn |
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| AdaBoost | scikit-learn |
| Bagging Classifier | scikit-learn |

---

## Web Interface

The prediction interface is built with **HTML** (frontend) and **Flask** (backend). Users fill out a short questionnaire and receive a depression risk prediction based on the loaded model.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/GregoiredCH/Depression-Prediction.git
cd Depression-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python treatment_v2.py
```

Then open the IP address shown in your terminal in a browser (e.g. `http://127.0.0.1:5000`).

---

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Flask
- pickle *(built-in)*

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask
```

---

> ⚠️ **Disclaimer:** This tool is a university/personal ML project and is **not a medical diagnosis tool**. If you or someone you know is struggling, please reach out to a healthcare professional.
