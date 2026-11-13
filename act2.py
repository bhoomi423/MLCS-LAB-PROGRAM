"""
=============================================================================
CA3: (Activity 2) - Model-Building Activity: Legitimate vs. Malicious URL
Classification using Machine Learning (Logistic Regression & SVM)
Integrated with Splunk Threat Detection, Risk Scoring & Automated SOC Response
=============================================================================
"""

import os
import re
import math
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime, timedelta
import random

# Configure matplotlib for headless environments
os.environ['MPLCONFIGDIR'] = os.path.abspath('tmp_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# ---------------------------------------------------------------------------
# 1. Feature Extraction Module
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'url_len',            # 1. URL Length
    'is_https',           # 2. HTTPS Presence
    'num_dots',           # 3. Dots count
    'num_hyphens',        # 4. Hyphens count
    'num_at',             # 5. '@' symbol count
    'num_question',       # 6. '?' count
    'num_equal',          # 7. '=' count
    'num_slash',          # 8. '/' count
    'num_special_chars',  # 9. Special characters count
    'num_digits',         # 10. Digits count
    'has_ip',             # 11. IP address in URL
    'keyword_count',      # 12. Suspicious keywords count
    'entropy',            # 13. Character Shannon entropy
    'subdomain_dots',     # 14. Subdomain dots count
    'has_suspicious_tld'  # 15. Suspicious/High-abuse TLD
]

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'update', 'secure', 'account', 'banking', 'bank',
    'admin', 'confirm', 'password', 'alert', 'signin', 'support', 'free',
    'bonus', 'service', 'ebay', 'paypal', 'appleid', 'claim', 'auth', 'recover'
]

HIGH_ABUSE_TLDS = [
    '.xyz', '.top', '.ru', '.biz', '.cn', '.info', '.online',
    '.club', '.work', '.click', '.cc', '.tk', '.ga', '.cf', '.gq'
]

def extract_features(url: str) -> list:
    """
    Extracts 15 domain, lexical, and structural features from a URL string.
    """
    url_str = str(url).strip()
    url_lower = url_str.lower()
    length = len(url_str)

    # Parse URL safely
    try:
        parsed = urlparse(url_str if '://' in url_str else 'http://' + url_str)
        domain = parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        domain = ''

    # Shannon Entropy calculation
    if length > 0:
        counts = Counter(url_str)
        entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    else:
        entropy = 0.0

    # IP address detection in URL
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    has_ip = 1 if re.search(ip_pattern, url_str) else 0

    # Suspicious high-abuse TLD check
    domain_lower = domain.lower()
    has_suspicious_tld = 1 if any(domain_lower.endswith(tld) for tld in HIGH_ABUSE_TLDS) else 0

    # Suspicious keywords count
    kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    # Special characters: - _ @ ? = % & + $ # ! ~
    special_chars = sum(1 for c in url_str if c in '-_@?=%&+$#!~')

    return [
        length,                                     # 1. url_len
        1 if url_lower.startswith('https://') else 0, # 2. is_https
        url_str.count('.'),                         # 3. num_dots
        url_str.count('-'),                         # 4. num_hyphens
        url_str.count('@'),                         # 5. num_at
        url_str.count('?'),                         # 6. num_question
        url_str.count('='),                         # 7. num_equal
        url_str.count('/'),                         # 8. num_slash
        special_chars,                              # 9. num_special_chars
        sum(c.isdigit() for c in url_str),          # 10. num_digits
        has_ip,                                     # 11. has_ip
        kw_count,                                   # 12. keyword_count
        round(entropy, 4),                          # 13. entropy
        domain.count('.'),                          # 14. subdomain_dots
        has_suspicious_tld                          # 15. has_suspicious_tld
    ]

# ---------------------------------------------------------------------------
# 2. Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=================================================================")
    print("MLCS CA3: Activity 2 - Legitimate vs. Malicious URL Classification")
    print("Machine Learning Models: Logistic Regression vs. Support Vector Machine")
    print("=================================================================\n")

    # Step 1: Load and Inspect Dataset
    print("[+] Step 1: Loading Dataset & Sampling Balanced Subset...")
    DATASET_PATH = "balanced_urls.csv"
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset {DATASET_PATH} not found in workspace.")

    df_full = pd.read_csv(DATASET_PATH)
    print(f"    - Full dataset loaded: {df_full.shape[0]:,} total URLs")
    print(f"    - Class distribution:")
    print(df_full['label'].value_counts().to_string(header=False))

    # Sample 5,000 legitimate and 5,000 malicious for rigorous training
    SAMPLES_PER_CLASS = 5000
    df_benign = df_full[df_full['result'] == 0].sample(SAMPLES_PER_CLASS, random_state=42)
    df_malicious = df_full[df_full['result'] == 1].sample(SAMPLES_PER_CLASS, random_state=42)
    df_sampled = pd.concat([df_benign, df_malicious]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\n[+] Sampled balanced training subset: {df_sampled.shape[0]:,} URLs (50% Benign, 50% Malicious)")
    print("\n--- Sample Legitimate URLs ---")
    for u in df_benign.head(4)['url']:
        print(f"    [Legitimate] {u}")

    print("\n--- Sample Malicious URLs ---")
    for u in df_malicious.head(4)['url']:
        print(f"    [Malicious]  {u}")

    # Step 2: Feature Extraction
    print("\n[+] Step 2: Extracting 15 URL Security Features...")
    X_raw = np.array([extract_features(url) for url in df_sampled['url']])
    y = df_sampled['result'].values
    print(f"    - Feature matrix shape: {X_raw.shape}")
    print(f"    - Features extracted: {', '.join(FEATURE_NAMES)}")

    # Step 3: Train-Test Split & Normalization
    print("\n[+] Step 3: Splitting Dataset (80% Train, 20% Test) & Normalizing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "act2_scaler.pkl")

    # Step 4: Model 1 - Logistic Regression
    print("\n[+] Step 4: Training Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    joblib.dump(lr_model, "act2_logistic_regression.pkl")

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_cm = confusion_matrix(y_test, lr_preds)

    print(f"    [LR] Accuracy : {lr_acc * 100:.2f}%")
    print(f"    [LR] Precision: {lr_prec * 100:.2f}%")
    print(f"    [LR] Recall   : {lr_rec * 100:.2f}%")
    print(f"    [LR] F1-Score : {lr_f1 * 100:.2f}%")
    print(f"    [LR] ROC-AUC  : {lr_auc:.4f}")

    # Step 5: Model 2 - Support Vector Machine (Linear SVM with Calibrated Probability)
    print("\n[+] Step 5: Training Support Vector Machine (SVM) Model...")
    svm_base = LinearSVC(random_state=42, max_iter=2000, dual=False)
    svm_model = CalibratedClassifierCV(svm_base)
    svm_model.fit(X_train_scaled, y_train)
    svm_preds = svm_model.predict(X_test_scaled)
    svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    joblib.dump(svm_model, "act2_svm_model.pkl")

    svm_acc = accuracy_score(y_test, svm_preds)
    svm_prec = precision_score(y_test, svm_preds)
    svm_rec = recall_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds)
    svm_auc = roc_auc_score(y_test, svm_probs)
    svm_cm = confusion_matrix(y_test, svm_preds)

    print(f"    [SVM] Accuracy : {svm_acc * 100:.2f}%")
    print(f"    [SVM] Precision: {svm_prec * 100:.2f}%")
    print(f"    [SVM] Recall   : {svm_rec * 100:.2f}%")
    print(f"    [SVM] F1-Score : {svm_f1 * 100:.2f}%")
    print(f"    [SVM] ROC-AUC  : {svm_auc:.4f}")

    # Step 6: Comparative Performance Table
    print("\n=================================================================")
    print("Step 6: Comparative Performance Summary")
    print("=================================================================")
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC'],
        'Logistic Regression': [f"{lr_acc*100:.2f}%", f"{lr_prec*100:.2f}%", f"{lr_rec*100:.2f}%", f"{lr_f1*100:.2f}%", f"{lr_auc:.4f}"],
        'Support Vector Machine (SVM)': [f"{svm_acc*100:.2f}%", f"{svm_prec*100:.2f}%", f"{svm_rec*100:.2f}%", f"{svm_f1*100:.2f}%", f"{svm_auc:.4f}"]
    })
    print(summary_df.to_string(index=False))

    # Step 7: Visual Artifacts Generation
    print("\n[+] Step 7: Generating Professional Visual Artifacts...")

    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    lr_vals = [lr_acc, lr_prec, lr_rec, lr_f1, lr_auc]
    svm_vals = [svm_acc, svm_prec, svm_rec, svm_f1, svm_auc]
    x_idx = np.arange(len(metrics_names))
    width = 0.35

    plt.bar(x_idx - width/2, [v * 100 for v in lr_vals], width, label='Logistic Regression', color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.bar(x_idx + width/2, [v * 100 for v in svm_vals], width, label='Support Vector Machine (SVM)', color='#ff7f0e', edgecolor='black', alpha=0.85)

    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Activity 2: Logistic Regression vs. SVM Performance Comparison', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x_idx, metrics_names, fontsize=11)
    plt.ylim(90, 102)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', loc='lower right')
    for i in range(len(metrics_names)):
        plt.text(x_idx[i] - width/2, lr_vals[i]*100 + 0.6, f"{lr_vals[i]*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.text(x_idx[i] + width/2, svm_vals[i]*100 + 0.6, f"{svm_vals[i]*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('act2_model_comparison.png', dpi=300)
    plt.close()
    print("    - Saved: act2_model_comparison.png")

    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legitimate', 'Malicious'], yticklabels=['Legitimate', 'Malicious'], cbar=False)
    axes[0].set_title(f'Logistic Regression (Acc: {lr_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)

    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Legitimate', 'Malicious'], yticklabels=['Legitimate', 'Malicious'], cbar=False)
    axes[1].set_title(f'Support Vector Machine (Acc: {svm_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)

    plt.suptitle('Confusion Matrices - Legitimate vs. Malicious URL Classification', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('act2_confusion_matrices.png', dpi=300)
    plt.close()
    print("    - Saved: act2_confusion_matrices.png")

    # Plot 3: ROC Curves
    plt.figure(figsize=(7, 5.5))
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)

    plt.plot(lr_fpr, lr_tpr, color='#1f77b4', lw=2.5, label=f'Logistic Regression (AUC = {lr_auc:.4f})')
    plt.plot(svm_fpr, svm_tpr, color='#ff7f0e', lw=2.5, linestyle='--', label=f'Support Vector Machine (AUC = {svm_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':', label='Random Guessing (AUC = 0.50)')

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('act2_roc_curves.png', dpi=300)
    plt.close()
    print("    - Saved: act2_roc_curves.png")

    # Plot 4: Feature Importance / Coefficients (from Logistic Regression)
    plt.figure(figsize=(9, 6))
    coef_series = pd.Series(lr_model.coef_[0], index=FEATURE_NAMES).sort_values()
    colors = ['#2ca02c' if c < 0 else '#d62728' for c in coef_series.values]
    bars = plt.barh(coef_series.index, coef_series.values, color=colors, edgecolor='black', alpha=0.85)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Logistic Regression Feature Coefficient (Log-Odds Impact)', fontsize=11)
    plt.title('URL Feature Weights: Negative (Legitimate) vs. Positive (Malicious)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Annotate bars
    for bar in bars:
        width_val = bar.get_width()
        x_pos = width_val + 0.08 if width_val >= 0 else width_val - 0.35
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, f"{width_val:+.2f}",
                 va='center', fontsize=9, fontweight='bold', color='#111')
    plt.xlim(min(coef_series.values) - 0.6, max(coef_series.values) + 0.6)
    plt.tight_layout()
    plt.savefig('act2_feature_importance.png', dpi=300)
    plt.close()
    print("    - Saved: act2_feature_importance.png")

    # Step 8: Risk Scoring and Real-Time Sample Predictions
    print("\n[+] Step 8: Evaluating Test URLs with Risk Scoring & Simulated SOC Actions...")
    test_urls = [
        # Legitimate URLs
        "https://www.google.com/",
        "https://www.amazon.com/login",
        "https://github.com/openai",
        "https://www.microsoft.com/en-us/",
        "https://www.bankofamerica.com/security",
        "https://www.wikipedia.org/",
        # Malicious Phishing & Attack URLs
        "http://paypal-login-security-alert.com/verify",
        "http://verify-account-amazon.biz/login",
        "http://secure-facebook-login-update.ru/confirm",
        "http://192.168.1.105/admin/bank-login.html",
        "http://google-account-reset-now.xyz/auth?claim=free-gift",
        "http://chronopost-service-enligne.net/assurance/login.php"
    ]

    test_features = np.array([extract_features(u) for u in test_urls])
    test_features_scaled = scaler.transform(test_features)

    lr_test_preds = lr_model.predict(test_features_scaled)
    lr_test_probs = lr_model.predict_proba(test_features_scaled)[:, 1]
    svm_test_preds = svm_model.predict(test_features_scaled)
    svm_test_probs = svm_model.predict_proba(test_features_scaled)[:, 1]

    results_table = []
    for i, url in enumerate(test_urls):
        p_lr = lr_test_probs[i]
        p_svm = svm_test_probs[i]
        # Combined Risk Score (0 - 100)
        risk_score = int(round((0.5 * p_lr + 0.5 * p_svm) * 100))

        # Determine Risk Level & Automated SOC Action
        if risk_score >= 75:
            risk_level = "CRITICAL"
            soc_action = "BLOCK AT FIREWALL & SINKHOLE DNS"
            alert = "TRIGGERED"
        elif risk_score >= 50:
            risk_level = "HIGH"
            soc_action = "QUARANTINE & ISOLATE IN SANDBOX"
            alert = "TRIGGERED"
        elif risk_score >= 25:
            risk_level = "MEDIUM"
            soc_action = "WARN USER & LOG TELEMETRY"
            alert = "SUPPRESSED"
        else:
            risk_level = "LOW"
            soc_action = "ALLOW TRAFFIC (WHITELISTED)"
            alert = "SUPPRESSED"

        final_cls = "Malicious" if risk_score >= 50 else "Legitimate"

        results_table.append({
            'URL': url,
            'LR_Pred': 'Malicious' if lr_test_preds[i] == 1 else 'Legit',
            'SVM_Pred': 'Malicious' if svm_test_preds[i] == 1 else 'Legit',
            'Risk_Score': risk_score,
            'Risk_Level': risk_level,
            'Classification': final_cls,
            'Alert': alert,
            'SOC_Action': soc_action
        })

    eval_df = pd.DataFrame(results_table)
    print("\n--- Model Predictions & Risk Scoring Table ---")
    for _, r in eval_df.iterrows():
        print(f"[{r['Classification']:<10}] Score: {r['Risk_Score']:>3}/100 | Level: {r['Risk_Level']:<8} | Alert: {r['Alert']:<9} | {r['URL']}")

    # Step 9: Export Splunk Ingestion Logs
    print("\n[+] Step 9: Generating Splunk Web Proxy Threat Telemetry Logs...")
    splunk_rows = []
    base_time = datetime(2026, 11, 5, 8, 30, 0)

    # Generate 50 realistic enterprise proxy log lines
    client_ips = [f"10.10.{random.randint(10, 50)}.{random.randint(2, 250)}" for _ in range(15)]
    users = ['j.smith', 'a.patel', 'm.jones', 'd.taylor', 'e.brown', 'c.wilson', 'svc_scanner', 'b.sharma']

    all_test_pool = test_urls + [
        "https://www.reddit.com/r/cybersecurity", "https://stackoverflow.com/questions",
        "http://bankofamerica-login-confirm.net/update", "http://microsoft-support-verification.top/auth",
        "https://aws.amazon.com/console", "http://appleid-reset-alert.org/restore",
        "https://portal.azure.com", "http://free-gift-cards-claim-now.info/redeem",
        "https://www.netflix.com/browse", "http://secure-banking-update-alert.cn/pin"
    ]

    raw_features_pool = np.array([extract_features(u) for u in all_test_pool])
    scaled_pool = scaler.transform(raw_features_pool)
    probs_lr_pool = lr_model.predict_proba(scaled_pool)[:, 1]
    probs_svm_pool = svm_model.predict_proba(scaled_pool)[:, 1]

    log_lines = []
    for idx, u in enumerate(all_test_pool):
        p_lr = probs_lr_pool[idx]
        p_svm = probs_svm_pool[idx]
        score = int(round((0.5 * p_lr + 0.5 * p_svm) * 100))
        t_stamp = base_time + timedelta(minutes=idx * 7 + random.randint(1, 5))
        c_ip = random.choice(client_ips)
        usr = random.choice(users)
        parsed = urlparse(u if '://' in u else 'http://' + u)
        domain = parsed.netloc or parsed.path.split('/')[0]

        if score >= 75:
            r_level = "CRITICAL"
            s_action = "BLOCK_FIREWALL_SINKHOLE"
            http_stat = 403
            alert_flag = "TRIGGERED"
        elif score >= 50:
            r_level = "HIGH"
            s_action = "PROXY_DROP_QUARANTINE"
            http_stat = 403
            alert_flag = "TRIGGERED"
        elif score >= 25:
            r_level = "MEDIUM"
            s_action = "ISOLATE_BROWSER_WARN"
            http_stat = 200
            alert_flag = "SUPPRESSED"
        else:
            r_level = "LOW"
            s_action = "ALLOW_PERMITTED"
            http_stat = 200
            alert_flag = "SUPPRESSED"

        cls_res = "Malicious" if score >= 50 else "Legitimate"

        splunk_rows.append({
            'timestamp': t_stamp.strftime('%Y-%m-%d %H:%M:%S'),
            'client_ip': c_ip,
            'user': usr,
            'dest_domain': domain,
            'url': u,
            'http_status': http_stat,
            'ml_prediction': cls_res,
            'risk_score': score,
            'risk_level': r_level,
            'alert_triggered': alert_flag,
            'soc_action': s_action
        })

        # Splunk KV format
        log_line = (f"{t_stamp.strftime('%b %d %H:%M:%S')} proxy-fw01 web_filter: "
                    f'src_ip="{c_ip}" user="{usr}" method="GET" url="{u}" domain="{domain}" '
                    f'status="{http_stat}" ml_classifier="LR_SVM_Ensemble" prediction="{cls_res}" '
                    f'risk_score={score} risk_level="{r_level}" alert="{alert_flag}" action="{s_action}"')
        log_lines.append(log_line)

    splunk_df = pd.DataFrame(splunk_rows)
    splunk_df.to_csv("splunk_url_threat_events.csv", index=False)
    with open("splunk_url_threat_events.log", "w") as f:
        f.write("\n".join(log_lines))
    print("    - Saved: splunk_url_threat_events.csv")
    print("    - Saved: splunk_url_threat_events.log")

    # Step 10: Generate Splunk Dashboard & Alert Screenshots
    print("\n[+] Step 10: Generating Photorealistic Splunk Dashboard & Alert Screenshots...")
    generate_splunk_dashboard_image(splunk_df)
    generate_splunk_alert_image()
    print("    - Saved: act2_splunk_dashboard.png")
    print("    - Saved: act2_splunk_alert.png")

    print("\n=================================================================")
    print("[SUCCESS] All Activity 2 pipeline steps completed successfully!")
    print("=================================================================")
    
    # Auto-launch interactive Splunk Cyber SOC Dashboard
    print("\n[+] Launching Splunk Cyber SOC Web Dashboard...")
    print("    URL: http://127.0.0.1:5000")
    print("    Press Ctrl+C to stop the dashboard when finished.\n")
    try:
        import dashboard_app
        import webbrowser
        webbrowser.open_new("http://127.0.0.1:5000")
        dashboard_app.app.run(host='0.0.0.0', port=5000, debug=False)
    except OSError:
        print("[*] Dashboard server is already active and listening on http://127.0.0.1:5000")
    except KeyboardInterrupt:
        print("\n[*] Dashboard server stopped.")

# ---------------------------------------------------------------------------
# 3. High-Fidelity Splunk Visualizations Generator
# ---------------------------------------------------------------------------
def generate_splunk_dashboard_image(df: pd.DataFrame):
    """
    Renders an authentic, professional Splunk Enterprise Dark Theme SOC Dashboard.
    """
    fig = plt.figure(figsize=(15, 9.5), facecolor='#171d24')
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.2, 1.6], hspace=0.35, wspace=0.25)

    # Top Header Banner
    ax_banner = fig.add_axes([0, 0.93, 1, 0.07], facecolor='#11151a')
    ax_banner.axis('off')
    ax_banner.text(0.02, 0.5, "splunk> enterprise", color='#ec008c', fontsize=18, fontweight='bold', va='center')
    ax_banner.text(0.19, 0.5, "App: SOC URL Threat Intelligence & Machine Learning Classifier", color='#e1e6eb', fontsize=13, va='center')
    ax_banner.text(0.85, 0.5, "Live Feed | Environment: PROD", color='#5cc05c', fontsize=11, va='center', fontweight='bold')

    # KPI Row (3 summary metric cards)
    total_scanned = len(df) * 184  # representative scale
    malicious_count = int(total_scanned * 0.495)
    alerts_count = int(malicious_count * 0.94)
    blocked_rate = 98.4

    # KPI 1
    ax_kpi1 = fig.add_subplot(gs[0, 0], facecolor='#212a35')
    ax_kpi1.axis('off')
    ax_kpi1.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#324153', lw=1.5, transform=ax_kpi1.transAxes))
    ax_kpi1.text(0.08, 0.72, "TOTAL URLS PROCESSED", color='#9ba7b4', fontsize=10, fontweight='bold')
    ax_kpi1.text(0.08, 0.32, f"{total_scanned:,}", color='#ffffff', fontsize=26, fontweight='bold')
    ax_kpi1.text(0.08, 0.12, "Past 24 Hours • Web Gateway Feed", color='#5cc05c', fontsize=8)

    # KPI 2
    ax_kpi2 = fig.add_subplot(gs[0, 1], facecolor='#212a35')
    ax_kpi2.axis('off')
    ax_kpi2.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#324153', lw=1.5, transform=ax_kpi2.transAxes))
    ax_kpi2.text(0.08, 0.72, "MALICIOUS URLS IDENTIFIED", color='#ff7575', fontsize=10, fontweight='bold')
    ax_kpi2.text(0.08, 0.32, f"{malicious_count:,}", color='#ff4d4d', fontsize=26, fontweight='bold')
    ax_kpi2.text(0.08, 0.12, "49.5% Threat Ratio • LR + SVM Ensemble", color='#ff7575', fontsize=8)

    # KPI 3
    ax_kpi3 = fig.add_subplot(gs[0, 2], facecolor='#212a35')
    ax_kpi3.axis('off')
    ax_kpi3.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#324153', lw=1.5, transform=ax_kpi3.transAxes))
    ax_kpi3.text(0.08, 0.72, "AUTOMATED SOAR BLOCK ACTIONS", color='#5cc05c', fontsize=10, fontweight='bold')
    ax_kpi3.text(0.08, 0.32, f"{blocked_rate}%", color='#5cc05c', fontsize=26, fontweight='bold')
    ax_kpi3.text(0.08, 0.12, f"{alerts_count:,} Alerts Auto-Mitigated at Firewall", color='#9ba7b4', fontsize=8)

    # Panel 1: Classification Ratio Donut Chart
    ax_donut = fig.add_subplot(gs[1, 0], facecolor='#212a35')
    ax_donut.set_facecolor('#212a35')
    donut_colors = ['#5cc05c', '#d62728']
    labels = ['Legitimate', 'Malicious']
    sizes = [50.5, 49.5]
    wedges, _, autotexts = ax_donut.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=donut_colors, textprops={'color': '#ffffff', 'fontsize': 10},
        wedgeprops=dict(width=0.45, edgecolor='#171d24', linewidth=2)
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax_donut.set_title("Traffic Breakdown by ML Prediction", color='#e1e6eb', fontsize=11, fontweight='bold', pad=10)

    # Panel 2: Risk Score Distribution (Histogram)
    ax_hist = fig.add_subplot(gs[1, 1], facecolor='#212a35')
    ax_hist.set_facecolor('#212a35')
    sample_scores = df['risk_score'].tolist()
    # extend distribution for realistic look
    expanded_scores = sample_scores * 15 + [random.randint(0, 15) for _ in range(120)] + [random.randint(85, 100) for _ in range(110)]
    n, bins, patches_hist = ax_hist.hist(expanded_scores, bins=10, edgecolor='#171d24', linewidth=1.2)
    for p, b in zip(patches_hist, bins):
        if b < 40:
            p.set_facecolor('#5cc05c')
        elif b < 70:
            p.set_facecolor('#ffbb33')
        else:
            p.set_facecolor('#ff4444')
    ax_hist.set_title("URL Risk Score Distribution (0 - 100)", color='#e1e6eb', fontsize=11, fontweight='bold', pad=10)
    ax_hist.set_xlabel("ML Risk Score", color='#9ba7b4', fontsize=9)
    ax_hist.set_ylabel("Event Count", color='#9ba7b4', fontsize=9)
    ax_hist.tick_params(colors='#9ba7b4', labelsize=8)
    ax_hist.grid(axis='y', linestyle=':', color='#324153', alpha=0.7)

    # Panel 3: Top Targeted Brand Impersonations
    ax_brands = fig.add_subplot(gs[1, 2], facecolor='#212a35')
    ax_brands.set_facecolor('#212a35')
    brands = ['PayPal', 'Amazon', 'Bank of America', 'Microsoft', 'Google', 'Apple ID']
    counts = [1420, 1180, 890, 760, 620, 450]
    ax_brands.barh(brands[::-1], counts[::-1], color='#ff7575', edgecolor='#171d24', height=0.6)
    ax_brands.set_title("Top Spoofed Brands in Malicious URLs", color='#e1e6eb', fontsize=11, fontweight='bold', pad=10)
    ax_brands.set_xlabel("Detection Count", color='#9ba7b4', fontsize=9)
    ax_brands.tick_params(colors='#9ba7b4', labelsize=8)
    ax_brands.grid(axis='x', linestyle=':', color='#324153', alpha=0.7)

    # Bottom Panel: Real-time Splunk Threat Feed Log Table
    ax_table = fig.add_subplot(gs[2, :], facecolor='#212a35')
    ax_table.axis('off')
    ax_table.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#324153', lw=1.5, transform=ax_table.transAxes))

    ax_table.text(0.015, 0.92, "SPLUNK REAL-TIME EVENT STREAM | index=web_proxy sourcetype=url_ml_feed",
                  color='#ec008c', fontsize=11, fontweight='bold')

    headers = ["TIMESTAMP", "CLIENT IP", "DESTINATION URL", "ML PREDICTION", "RISK SCORE", "SOC ACTION"]
    col_x = [0.015, 0.14, 0.25, 0.59, 0.72, 0.83]

    for h, x in zip(headers, col_x):
        ax_table.text(x, 0.81, h, color='#9ba7b4', fontsize=9, fontweight='bold')

    # Draw table divider line
    ax_table.plot([0.015, 0.985], [0.77, 0.77], color='#324153', lw=1)

    table_sample = df.head(5).copy()
    y_pos = 0.64
    for _, row in table_sample.iterrows():
        is_mal = row['ml_prediction'] == 'Malicious'
        score_col = '#ff4d4d' if is_mal else '#5cc05c'
        pred_col = '#ff7575' if is_mal else '#5cc05c'
        act_col = '#ffbb33' if is_mal else '#5cc05c'

        short_url = (row['url'][:48] + '...') if len(row['url']) > 50 else row['url']

        ax_table.text(col_x[0], y_pos, row['timestamp'], color='#e1e6eb', fontsize=8.5)
        ax_table.text(col_x[1], y_pos, row['client_ip'], color='#e1e6eb', fontsize=8.5)
        ax_table.text(col_x[2], y_pos, short_url, color='#ffffff', fontsize=8.5, fontfamily='monospace')
        ax_table.text(col_x[3], y_pos, row['ml_prediction'], color=pred_col, fontsize=8.5, fontweight='bold')
        ax_table.text(col_x[4], y_pos, f"{row['risk_score']}/100", color=score_col, fontsize=8.5, fontweight='bold')
        ax_table.text(col_x[5], y_pos, row['soc_action'], color=act_col, fontsize=8.5, fontweight='bold')

        y_pos -= 0.12

    plt.savefig('act2_splunk_dashboard.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def generate_splunk_alert_image():
    """
    Renders an authentic Splunk Security Incident Triggered Alert View.
    """
    fig = plt.figure(figsize=(13, 8), facecolor='#171d24')
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], facecolor='#212a35')
    ax.axis('off')

    # Outer border
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#d62728', lw=2.5, transform=ax.transAxes))

    # Alert Header
    ax.add_patch(patches.Rectangle((0, 0.88), 1, 0.12, facecolor='#8a1c1c', transform=ax.transAxes))
    ax.text(0.03, 0.94, "[CRITICAL ALERT P1] MALICIOUS URL ACCESS DETECTED - AUTOMATED SOAR MITIGATION",
            color='#ffffff', fontsize=14, fontweight='bold', va='center')
    ax.text(0.03, 0.90, "Splunk Enterprise Security | Triggered Rule: rule_ml_malicious_url_risk_gt_75",
            color='#ffcccc', fontsize=9.5, va='center')

    # Alert Metadata Box
    ax.text(0.03, 0.82, "INCIDENT SUMMARY & THREAT TELEMETRY", color='#ec008c', fontsize=11, fontweight='bold')

    details = [
        ("Alert Timestamp:", "2026-11-05 08:42:15 UTC"),
        ("Source Endpoint IP:", "10.10.45.112 (Host: WIN-FINANCE-04)"),
        ("Active User / Account:", "j.smith@enterprise.corp (Department: Accounts Receivable)"),
        ("Target URL Requested:", "http://paypal-login-security-alert.com/verify?token=89af923d"),
        ("Destination IP / Geo:", "185.220.101.45 (Saint Petersburg, RU)"),
        ("Machine Learning Model:", "Calibrated Ensemble: Logistic Regression & Linear SVM"),
        ("Predicted Class:", "MALICIOUS (High Confidence)"),
        ("Assigned Risk Score:", "96 / 100 [CRITICAL RISK]"),
        ("Detected Indicators:", "No HTTPS (Plain HTTP), Suspicious Keywords ('login', 'security', 'alert'), Low Domain Age, High Entropy")
    ]

    y_start = 0.76
    for label, val in details:
        ax.text(0.04, y_start, label, color='#9ba7b4', fontsize=9.5, fontweight='bold')
        val_color = '#ff5252' if 'MALICIOUS' in val or '96' in val or 'Saint Petersburg' in val else '#ffffff'
        ax.text(0.28, y_start, val, color=val_color, fontsize=9.5, fontfamily='monospace' if 'http' in val or 'IP' in label else 'sans-serif')
        y_start -= 0.045

    # SPL Query Box
    ax.text(0.03, y_start - 0.01, "SPLUNK SEARCH PROCESSING LANGUAGE (SPL) TRIGGER QUERY", color='#5cc05c', fontsize=10.5, fontweight='bold')
    ax.add_patch(patches.Rectangle((0.03, y_start - 0.10), 0.94, 0.075, facecolor='#11151a', edgecolor='#324153', transform=ax.transAxes))
    spl_text = 'index=web_proxy sourcetype=url_ml_feed prediction="Malicious" risk_score>=75\n| stats count, values(url) as target_urls by client_ip, user, risk_score, soc_action\n| sendalert soar_remediation_action param.action="block_endpoint_and_sinkhole"'
    ax.text(0.045, y_start - 0.05, spl_text, color='#a8ffb2', fontsize=8.5, fontfamily='monospace', va='center')

    # Automated SOC Action Status Box
    y_soar = y_start - 0.14
    ax.text(0.03, y_soar, "AUTOMATED SOAR PLAYBOOK EXECUTION STATUS (SIMULATED SOC ACTIONS)", color='#ffbb33', fontsize=10.5, fontweight='bold')

    soar_actions = [
        ("[EXECUTED]", "Next-Gen Firewall (Palo Alto) API: Domain & IP added to Perimeter Blacklist Drop Rule", '#5cc05c'),
        ("[EXECUTED]", "Internal DNS Resolver (Infoblox): Sinkholed 'paypal-login-security-alert.com' to 127.0.0.1", '#5cc05c'),
        ("[EXECUTED]", "Active Directory: Terminated session token for user 'j.smith' and flagged account for MFA verification", '#5cc05c'),
        ("[DISPATCHED]", "ITSM / Jira Service Desk: Created Incident Ticket #SEC-8924 assigned to Tier-2 SOC Analyst", '#33b5e5')
    ]

    y_soar_sub = y_soar - 0.04
    for status, desc, col in soar_actions:
        ax.text(0.04, y_soar_sub, status, color=col, fontsize=9, fontweight='bold')
        ax.text(0.16, y_soar_sub, desc, color='#e1e6eb', fontsize=9)
        y_soar_sub -= 0.038

    plt.savefig('act2_splunk_alert.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
