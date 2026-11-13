# MLCS LAB PROGRAM

This repository contains the Machine Learning and Cyber Security (MLCS) laboratory experiments, ranging from **Experiment 1 to Experiment 12**, along with dataset utilities, trained model artifacts, and network traffic anomaly analysis scripts.

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
| **Exp 7** | CAPTCHA Solving Model | Solves noisy CAPTCHA challenges by extracting pixel intensity features from synthetic images. | Synthetic CAPTCHA Generator + Random Forest | `exp7_confusion_matrix.png`, `exp7_captcha_model.pkl` |
| **Exp 8** | Email Spoofing Pattern Detection | Detects email spoofing patterns using email authentication indicators (SPF, DKIM, DMARC, domain alignment). | Random Forest Classifier | `exp8_confusion_matrix.png`, `email_spoofing.csv` |
| **Exp 9** | Email Spam Text Classification | Classifies email text as spam or ham using Bag-of-Words and probabilistic text modeling. | CountVectorizer + Multinomial Naive Bayes | `exp9_confusion_matrix.png`, `email_spam.csv` |
| **Exp 10** | K-Means Network Event Clustering | Unsupervised clustering of network events into Normal and Abnormal (attack/anomaly) categories. | StandardScaler + K-Means Clustering ($K=2$) | `exp10_kmeans_network_events.png`, `network_events.csv` |
| **Exp 11** | Database Malicious Injection Detection | Detects malicious SQL injection (SQLi) attacks within database queries using rule-based decision trees. | Feature Extraction + Decision Tree Classifier | `exp11_confusion_matrix.png`, `exp11_decision_tree.png`, `sql_injection_dataset.csv` |
| **Exp 12** | WSN Context-Aware Anomaly Detection | Utilizes contextual features (hour, source IP, event type, transmission rate, RSSI) to identify malicious wireless sensor network events. | Context Encoding + Random Forest Classifier | `exp12_confusion_matrix.png`, `exp12_feature_importance.png`, `wsn_context_events.csv` |
| **Activity 1** | DDoS Time Series Anomaly Detection | Rolling 3-sigma dynamic thresholding on network traffic payload logs to identify volumetric attacks. | 3-Sigma Anomaly Thresholding | `activity1.py`, `activity1_report.md`, `act_traffic_anomaly_plot_activity.png` |
| **Activity 2** | URL Threat Classification & SOC Dashboard | Lexical and structural URL feature extraction, Logistic Regression & SVM classification, risk scoring, Splunk SOC threat dashboard integration, and interactive web dashboard. | Logistic Regression + Calibrated Linear SVM | `act2.py` / `activity2.py`, `dashboard_app.py`, `run_dashboard.bat`, `act2_report.md`, `act2_*.png` |

---

## 🚀 Activity 2 Interactive Demonstration Dashboard

To run the interactive demonstration dashboard for **Activity 2**:

```bash
# Option 1: Double click the Windows batch launcher
run_dashboard.bat

# Option 2: Run via Python directly
python run_dashboard.py
# or
python dashboard_app.py
```

The web dashboard starts at **`http://127.0.0.1:5000`** and opens automatically in your browser.

### Key Interactive Features for Practical Demonstration:
1. **Live URL Threat Scanner**: Test custom URLs or select 1-click preset test cases (PayPal phishing lure, raw IPv4 C2, `.xyz` social engineering, legitimate Google, Bank of America, GitHub). Computes real-time risk scores (0–100), alert triggers, gateway actions (HTTP 403 / 200), and automated SOAR playbooks.
2. **15 Extracted Features Security Matrix**: Live table showing length, HTTPS presence, dots, hyphens, `@`, `?`, `=`, `/`, digits, IP host flag, Shannon entropy, and high-abuse TLD flag with threat rationale.
3. **SOC Threat Telemetry & Splunk KPIs**: Executive metric cards (Total Scanned, Malicious Count, 98.4% Mitigation Rate), interactive Chart.js doughnut traffic breakdown, and top spoofed brands horizontal bar chart.
4. **Model Performance & Diagnostics**: Side-by-side comparative table (Logistic Regression `98.15%` vs. SVM `98.10%`) and visual plots (ROC Curves, Confusion Matrices, Model Comparison, Feature Importance).
5. **Splunk Proxy Event Logs & Live Simulation**: Filterable SIEM log stream with a **"⚡ Simulate Inbound Event"** button to inject live network events on demand.
6. **Splunk P1 Incident Alert & SPL Query**: Realistic critical incident alert format with automated SOAR playbooks (Firewall rule injection, DNS sinkhole, Active Directory session revocation, ITSM Jira ticket).
7. **Batch URL Threat Audit**: Audit dozens of URLs simultaneously and export scored threat telemetry to CSV.

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
├── exp7.py                            # Experiment 7: CAPTCHA Solving Model (Random Forest Classifier)
├── exp8.py                            # Experiment 8: Email Spoofing Detection (Random Forest)
├── exp9.py                            # Experiment 9: Email Spam Text Classification (Naive Bayes)
├── exp10.py                           # Experiment 10: K-Means Network Event Clustering
├── exp11.py                           # Experiment 11: Database Malicious Injection Detection (Decision Tree)
├── exp12.py                           # Experiment 12: Context-Aware WSN Malicious Event Detection
├── activity1.py                       # Extended Network Traffic Analysis & Anomaly Detection
├── ddos_timeseries_analysis.py        # DDoS Time Series Analysis Script
├── activity1_report.md                # Network Traffic Anomaly Analysis Report
├── act2.py                            # Activity 2: Legitimate vs. Malicious URL Classification & Splunk SOC
├── act2_report.md                     # Activity 2: Model-Building & Splunk Threat Detection Report
├── activity2.py                       # Activity 2: Alias / Full Script Name
├── activity2_report.md                # Activity 2: Alias / Full Report Document
│
├── DDoS_dataset.csv                   # Synthetic DDoS Time Series Network Traffic Data
├── cyber_attacks.csv                  # Network Flow Cyber Attack Dataset
├── urls_dataset.csv                   # Sample Phishing/Legitimate URL Dataset
├── Time-Series_Network_logs.csv       # High-granularity Network Logs
├── balanced_urls.csv                  # Large-scale Balanced URL Dataset
├── email_spoofing.csv                 # Email Spoofing Indicators Dataset
├── email_spam.csv                     # Email Spam / Ham Text Dataset
├── network_events.csv                 # Network Event Traffic Features Dataset
├── sql_injection_dataset.csv          # SQL Queries & Injection Attack Dataset
├── wsn_context_events.csv             # Wireless Sensor Network Contextual Events Dataset
│
├── *.png                              # Output Visualizations, Decision Trees & Confusion Matrices
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

#### Experiment 7: CAPTCHA Solver Model
```bash
python exp7.py
```

#### Experiment 8: Email Spoofing Pattern Detection
```bash
python exp8.py
```

#### Experiment 9: Email Spam Text Classification
```bash
python exp9.py
```

#### Experiment 10: K-Means Network Event Clustering
```bash
python exp10.py
```

#### Experiment 11: Database Malicious Injection Detection
```bash
python exp11.py
```

#### Experiment 12: Context-Aware WSN Malicious Event Detection
```bash
python exp12.py
```

---

## 📊 Summary of Models & Results

- **Time Series Anomaly Detection (Exp 1 & Exp 2)**: Effectively isolates transient DDoS traffic spikes from underlying normal network usage using decomposition and ARIMA forecasting.
- **Supervised Attack Classification (Exp 3)**: Random Forest ensemble model achieves high precision in categorizing attack vs normal network flows.
- **NLP & Lexical URL Classifiers (Exp 4, Exp 5, Exp 6)**: Multi-stage evaluation using TF-IDF n-grams (3-5 char length), Shannon entropy analysis, and SVM/Random Forest models to flag malicious domain patterns.
- **CAPTCHA Solving Model (Exp 7)**: Synthesizes noisy image data and uses Random Forest classification on pixel intensity features to accurately break CAPTCHA challenge strings.
- **Email Spoofing Detection (Exp 8)**: Leverages email authentication header features (SPF, DKIM, DMARC, sender/reply-to domain matches) to identify impersonation and spoofing patterns.
- **Email Spam Classification (Exp 9)**: Implements text tokenization with CountVectorizer and Multinomial Naive Bayes to classify legitimate and spam messages.
- **Unsupervised Anomaly Clustering (Exp 10)**: Uses K-Means ($K=2$) on scaled network flow metrics (duration, packets, bytes, failed logins) to separate normal traffic from attack spikes without labels.
- **Database Injection Detection (Exp 11)**: Extracts syntactic SQL features (quotes, comments, tautologies, keywords) and trains a human-interpretable Decision Tree to intercept SQLi attacks.
- **WSN Context Anomaly Detection (Exp 12)**: Encodes environmental and operational context (timestamp hour, source node IP, event type, packet rate, RSSI, energy drain) with Random Forest to detect anomalous sensor network behavior.

---

## 📜 License
This repository is created for educational and academic laboratory work.
