"""
=============================================================================
MLCS CA3 Activity 2: Demonstration Dashboard Launcher
Unified startup runner and demonstration guide for evaluation
=============================================================================
"""

import os
import sys
import subprocess

def check_files():
    required = [
        "act2_core.py",
        "dashboard_app.py",
        "act2_logistic_regression.pkl",
        "act2_svm_model.pkl",
        "act2_scaler.pkl",
        "templates/dashboard.html",
        "act2_model_comparison.png",
        "act2_confusion_matrices.png",
        "act2_roc_curves.png",
        "act2_feature_importance.png",
        "act2_splunk_dashboard.png",
        "act2_splunk_alert.png"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"[!] Warning: Missing expected files: {missing}")
        return False
    return True

def print_demo_guide():
    print("=========================================================================")
    print("  MLCS CA3 ACTIVITY 2: PRACTICAL DEMONSTRATION GUIDE")
    print("  Topic: Legitimate vs. Malicious URL Classification with Machine Learning")
    print("=========================================================================")
    print("""
During your demonstration/viva, present the project across these 5 stages:

1. 🎯 TAB 1: Live Interactive URL Threat Scanner
   - Click the preset test cases:
     * 'Credential Phishing (PayPal)': Demonstrates detection of hyphen chaining,
       plain HTTP, and 4 urgent lure keywords. (Result: 100/100 CRITICAL)
     * 'Direct IPv4 Host': Demonstrates detection of numeric IP bypassing DNS.
     * 'Legitimate Search (Google)' & 'Banking (BofA)': Shows zero false alarms.
   - Show the 15 Extracted Features table (Entropy, TLD, Special Chars, Digits).
   - Show the Automated SOAR Playbook execution steps (Firewall drop, DNS sinkhole).

2. 📊 TAB 2: SOC Threat Intelligence Telemetry & Splunk KPIs
   - Total Scanned (4,048), Malicious Count (2,004), Mitigation Rate (98.4%).
   - Interactive charts: Traffic breakdown donut and Top spoofed brands bar chart.
   - Show the official Splunk Enterprise Dark Theme SOC Dashboard graphic.

3. 🔬 TAB 3: Model Evaluation Diagnostics (LR vs. SVM)
   - Explain comparative results:
     * Logistic Regression: 98.15% Accuracy, 97.90% Precision, 98.40% Recall
     * Support Vector Machine: 98.10% Accuracy, 97.80% Precision, 98.40% Recall
     * Both models achieve ROC-AUC > 0.992!
   - Highlight the 4 plots: ROC Curves, Confusion Matrices, Feature Weights.

4. 📋 TAB 4: Splunk Event Stream & Live Simulation
   - Click '⚡ Simulate Inbound Event' to demonstrate real-time event ingestion.
   - Filter by Risk Level (Critical, High, Medium, Low).

5. 🚨 TAB 5: Splunk P1 Incident Alert & Batch Scanner
   - Show the SPL correlation query and triggered P1 security incident view.
   - Demonstrate bulk auditing by pasting URLs and exporting scored CSV reports!
=========================================================================
    """)

if __name__ == '__main__':
    print_demo_guide()
    if check_files():
        print("[✓] All model artifacts, visual plots, and templates verified.")
    print("\n[+] Starting Splunk Cyber SOC Dashboard on http://127.0.0.1:5000 ...")
    subprocess.run([sys.executable, "dashboard_app.py"])
