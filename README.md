# MLCS LAB PROGRAM

This repository contains the Machine Learning and Cyber Security (MLCS) laboratory experiments, ranging from **Experiment 1 to Experiment 6**, along with dataset utilities, trained model artifacts, and network traffic anomaly analysis scripts.

---

## 📌 Repository Overview

| Experiment | Title | Description | Model / Technique | Output Artifact |
|---|---|---|---|---|
| **Exp 1** | Time Series Decomposition | Decomposes network traffic into Trend, Seasonal, and Residual components to analyze DDoS spikes. | `statsmodels` (Additive Model) | `time_series_decomposition.png` |
| **Exp 2** | Traffic Forecasting & DDoS Detection | Fits an ARIMA model on time-series network data to set a baseline and flag statistical anomalies (DDoS attacks). | ARIMA `(2, 1, 2)` | `ddos_forecast_plot.png` |
| **Exp 3** | Cyber Attack Classification | Classifies network flow data into Normal vs. Attack traffic. | Random Forest Classifier | `cyber_attack_classification.png` |
| **Exp 4** | Phishing URL Detection | Detects phishing websites using character-level n-gram features from URL strings. | TF-IDF + Logistic Regression | `phishing_confusion_matrix.png`, `phishing_url_model.pkl` |
| **Exp 5** | Malicious URL Classification | High-accuracy malicious URL identification trained on balanced URL datasets. | TF-IDF + Support Vector Machine (SVM) | `exp5_svm_confusion_matrix.png`, `exp5_svm_model.pkl` |
| **Exp 6** | URL Abnormal Pattern Detection | Custom feature extraction (Shannon entropy, URL/hostname length, IP patterns, keyword frequency) for anomaly detection. | Feature Engineering + Random Forest | `exp6_confusion_matrix.png`, `exp6_feature_importance.png`, `exp6_rf_model.pkl` |

---

## 📁 Directory Structure

```
MLCS LAB PROGRAM/
├── exp1.py                            # Experiment 1: Time Series Decomposition
├── exp2.py                            # Experiment 2: ARIMA Traffic Forecasting & DDoS Anomaly Detection
├── exp3.py                            # Experiment 3: Cyber Attack Classification (Random Forest)
├── exp4.py                            # Experiment 4: Phishing URL Detection (TF-IDF + Logistic Regression)
├── exp5.py                            # Experiment 5: Malicious URL Detection (SVM Classifier)
├── exp6.py                            # Experiment 6: URL Abnormal Pattern Detection (Feature Engineering)
├── activity1.py                       # Extended Network Traffic Analysis & Anomaly Detection
├── ddos_timeseries_analysis.py        # DDoS Time Series Analysis Script
├── activity1_report.md                # Network Traffic Anomaly Analysis Report
│
├── DDoS_dataset.csv                   # Synthetic DDoS Time Series Network Traffic Data
├── cyber_attacks.csv                  # Network Flow Cyber Attack Dataset
├── urls_dataset.csv                   # Sample Phishing/Legitimate URL Dataset
├── Time-Series_Network_logs.csv       # High-granularity Network Logs
├── balanced_urls.csv                  # Large-scale Balanced URL Dataset
│
├── *.png                              # Output Visualizations & Confusion Matrices
├── *.pkl                              # Trained Models & Vectorizers
├── requirements.txt                   # Python Dependencies
└── README.md                          # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.8+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/bhoomi423/MLCS-LAB-PROGRAM.git
cd MLCS-LAB-PROGRAM
pip install -r requirements.txt
```

### 2. Running the Experiments

#### Experiment 1: Time Series Decomposition
```bash
python exp1.py
```

#### Experiment 2: Network Traffic Forecasting & DDoS Detection
```bash
python exp2.py
```

#### Experiment 3: Cyber Attack Classification
```bash
python exp3.py
```

#### Experiment 4: Phishing URL Detection
```bash
python exp4.py
```

#### Experiment 5: SVM Malicious URL Classifier
```bash
python exp5.py
```

#### Experiment 6: Feature-Engineered URL Anomaly Detection
```bash
python exp6.py
```

---

## 📊 Summary of Models & Results

- **Time Series Anomaly Detection (Exp 1 & Exp 2)**: Effectively isolates transient DDoS traffic spikes from underlying normal network usage.
- **Supervised Attack Classification (Exp 3)**: Random Forest ensemble model achieves high precision in categorizing attack vs normal network flows.
- **NLP & Lexical URL Classifiers (Exp 4, Exp 5, Exp 6)**: Multi-stage evaluation using TF-IDF n-grams (3-5 char length), Shannon entropy analysis, and SVM/Random Forest models to flag malicious domain patterns.

---

## 📜 License
This repository is created for educational and academic laboratory work.
