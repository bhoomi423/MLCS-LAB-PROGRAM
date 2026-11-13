"""
MLCS CA3 Activity 2: Core ML Engine & Threat Intelligence Helper
Shared module for Model Inference, Feature Extraction, Risk Scoring & SOAR Playbooks
"""

import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime, timedelta
import random

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

FEATURE_DESCRIPTIONS = {
    'url_len': 'Total length of URL string (Phishing URLs often contain obfuscation tokens)',
    'is_https': 'Encrypted HTTPS protocol indicator (1 = True, 0 = Plain HTTP)',
    'num_dots': 'Count of dot "." characters (Nested subdomains used in camouflage)',
    'num_hyphens': 'Count of hyphens "-" (Common in typosquatting brand variations)',
    'num_at': 'Count of "@" symbol (Used to exploit URL authentication masking)',
    'num_question': 'Count of query delimiter "?"',
    'num_equal': 'Count of parameter separators "="',
    'num_slash': 'Count of directory depth forward slashes "/"',
    'num_special_chars': 'Special character density (-_@?=%&+$#!~)',
    'num_digits': 'Count of numeric digits (Correlated with DGA and session tokens)',
    'has_ip': 'Raw IPv4 address host instead of domain name (1 = IP detected, 0 = Domain)',
    'keyword_count': 'Count of high-threat social engineering keywords found',
    'entropy': 'Shannon character entropy (Randomness measure for DGA/encoded strings)',
    'subdomain_dots': 'Count of subdomains inside destination FQDN',
    'has_suspicious_tld': 'High-abuse Top-Level Domain flag (.xyz, .top, .ru, .biz, .cn, etc.)'
}

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'update', 'secure', 'account', 'banking', 'bank',
    'admin', 'confirm', 'password', 'alert', 'signin', 'support', 'free',
    'bonus', 'service', 'ebay', 'paypal', 'appleid', 'claim', 'auth', 'recover'
]

HIGH_ABUSE_TLDS = [
    '.xyz', '.top', '.ru', '.biz', '.cn', '.info', '.online',
    '.club', '.work', '.click', '.cc', '.tk', '.ga', '.cf', '.gq'
]

PRESET_DEMO_URLS = [
    {
        "category": "Credential Phishing (PayPal Lure)",
        "url": "http://paypal-login-security-alert.com/verify?token=89af923d",
        "expected": "Malicious",
        "note": "Plain HTTP, brand spoofing, hyphen chaining, urgent keywords ('login', 'security', 'alert', 'verify')"
    },
    {
        "category": "High-Abuse TLD Scam (Amazon Impersonation)",
        "url": "http://verify-account-amazon.biz/login",
        "expected": "Malicious",
        "note": ".biz abuse TLD, multiple keywords, unencrypted HTTP"
    },
    {
        "category": "Direct IP C2 Server (Raw IPv4)",
        "url": "http://192.168.1.105/admin/bank-login.html",
        "expected": "Malicious",
        "note": "Direct IPv4 host bypassing DNS reputation, 'admin' and 'bank' lure"
    },
    {
        "category": "Social Engineering Gift Scam (.xyz)",
        "url": "http://google-account-reset-now.xyz/auth?claim=free-gift",
        "expected": "Malicious",
        "note": "Suspicious .xyz TLD, 'claim=free-gift' lure parameter, multiple hyphens"
    },
    {
        "category": "Compromised Foreign TLD (.ru)",
        "url": "http://secure-facebook-login-update.ru/confirm",
        "expected": "Malicious",
        "note": ".ru high-abuse TLD, Facebook impersonation, plain HTTP"
    },
    {
        "category": "Legitimate Search Engine (Google)",
        "url": "https://www.google.com/",
        "expected": "Legitimate",
        "note": "Official HTTPS, clean root domain, zero phishing keywords"
    },
    {
        "category": "Legitimate Banking (Bank of America)",
        "url": "https://www.bankofamerica.com/security",
        "expected": "Legitimate",
        "note": "Official corporate domain with EV SSL, legitimate security advisory"
    },
    {
        "category": "Legitimate E-Commerce (Amazon)",
        "url": "https://www.amazon.com/login",
        "expected": "Legitimate",
        "note": "Official amazon.com portal, trusted TLS certificate"
    },
    {
        "category": "Legitimate Developer Portal (GitHub)",
        "url": "https://github.com/openai",
        "expected": "Legitimate",
        "note": "Clean hierarchy, standard GitHub organization repository"
    },
    {
        "category": "Legitimate Cloud Console (Microsoft Azure)",
        "url": "https://portal.azure.com",
        "expected": "Legitimate",
        "note": "Authorized Microsoft cloud enterprise management portal"
    }
]

# Cached models
_MODELS_CACHE = {}

def load_models(base_dir=None):
    """Loads and caches LR, SVM, and Scaler models."""
    global _MODELS_CACHE
    if 'lr' in _MODELS_CACHE:
        return _MODELS_CACHE['lr'], _MODELS_CACHE['svm'], _MODELS_CACHE['scaler']

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    lr_path = os.path.join(base_dir, 'act2_logistic_regression.pkl')
    svm_path = os.path.join(base_dir, 'act2_svm_model.pkl')
    scaler_path = os.path.join(base_dir, 'act2_scaler.pkl')

    if not (os.path.exists(lr_path) and os.path.exists(svm_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(f"Trained model artifacts not found in {base_dir}")

    lr = joblib.load(lr_path)
    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)

    _MODELS_CACHE['lr'] = lr
    _MODELS_CACHE['svm'] = svm
    _MODELS_CACHE['scaler'] = scaler

    return lr, svm, scaler


def extract_features(url: str) -> list:
    """Extracts 15 domain, lexical, and structural features from a URL string."""
    url_str = str(url).strip()
    # Normalize scheme if omitted by user (e.g. 'google.com' -> 'https://google.com')
    if not (url_str.lower().startswith('http://') or url_str.lower().startswith('https://')):
        url_str = 'https://' + url_str

    url_lower = url_str.lower()
    length = len(url_str)

    try:
        parsed = urlparse(url_str if '://' in url_str else 'http://' + url_str)
        domain = parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        domain = ''

    if length > 0:
        counts = Counter(url_str)
        entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    else:
        entropy = 0.0

    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    has_ip = 1 if re.search(ip_pattern, url_str) else 0

    domain_lower = domain.lower()
    has_suspicious_tld = 1 if any(domain_lower.endswith(tld) for tld in HIGH_ABUSE_TLDS) else 0

    kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
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


def extract_features_dict(url: str) -> dict:
    """Extracts features and packages them with names and descriptions."""
    vals = extract_features(url)
    res = {}
    for name, val in zip(FEATURE_NAMES, vals):
        res[name] = {
            'value': val,
            'description': FEATURE_DESCRIPTIONS.get(name, '')
        }
    return res


def analyze_url(url: str, models=None) -> dict:
    """
    Performs full classification, probability estimation, risk scoring,
    and automated SOAR response generation for a single URL.
    """
    if models is None:
        lr, svm, scaler = load_models()
    else:
        lr, svm, scaler = models

    features_raw = extract_features(url)
    features_array = np.array([features_raw])
    features_scaled = scaler.transform(features_array)

    # Predictions and calibrated probabilities
    lr_pred = int(lr.predict(features_scaled)[0])
    lr_prob = float(lr.predict_proba(features_scaled)[0][1])

    svm_pred = int(svm.predict(features_scaled)[0])
    svm_prob = float(svm.predict_proba(features_scaled)[0][1])

    # Ensemble Risk Score (0-100)
    risk_score = int(round((0.5 * lr_prob + 0.5 * svm_prob) * 100))

    # Tiers and Automated SOAR Responses
    if risk_score >= 75:
        risk_level = "CRITICAL"
        risk_color = "#e53e3e"  # Red
        soc_action = "BLOCK AT FIREWALL & SINKHOLE DNS"
        http_status = 403
        alert_triggered = "TRIGGERED"
        alert_severity = "P1 - Critical Security Alert"
        soar_playbook = [
            {"step": 1, "component": "Next-Gen Firewall (Palo Alto / Fortinet)", "action": "Add destination host & IP to Edge Perimeter Drop Rule", "status": "EXECUTED", "badge": "bg-red-900 text-red-200"},
            {"step": 2, "component": "Internal DNS Resolver (Infoblox)", "action": "Sinkhole malicious FQDN to safe loopback (127.0.0.1)", "status": "EXECUTED", "badge": "bg-red-900 text-red-200"},
            {"step": 3, "component": "Identity Provider / Active Directory", "action": "Revoke active Kerberos/OAuth session token for endpoint", "status": "EXECUTED", "badge": "bg-yellow-900 text-yellow-200"},
            {"step": 4, "component": "ITSM Ticketing (Jira / ServiceNow)", "action": f"Dispatched Incident #SEC-{random.randint(8100, 9999)} to Tier-2 SOC Analyst", "status": "DISPATCHED", "badge": "bg-blue-900 text-blue-200"}
        ]
    elif risk_score >= 50:
        risk_level = "HIGH"
        risk_color = "#ed8936"  # Orange
        soc_action = "QUARANTINE & ISOLATE IN SANDBOX"
        http_status = 403
        alert_triggered = "TRIGGERED"
        alert_severity = "P2 - High Threat Alert"
        soar_playbook = [
            {"step": 1, "component": "Web Application Proxy", "action": "HTTP 403 Forbidden page served to client workstation", "status": "EXECUTED", "badge": "bg-orange-900 text-orange-200"},
            {"step": 2, "component": "Sandbox Detonation (Cuckoo)", "action": "Quarantine target payload in isolated detonation environment", "status": "IN-PROGRESS", "badge": "bg-yellow-900 text-yellow-200"},
            {"step": 3, "component": "Endpoint Detection (EDR)", "action": "Initiate full telemetry snapshot of requesting browser process", "status": "EXECUTED", "badge": "bg-blue-900 text-blue-200"}
        ]
    elif risk_score >= 25:
        risk_level = "MEDIUM"
        risk_color = "#ecc94b"  # Yellow
        soc_action = "WARN USER & LOG TELEMETRY"
        http_status = 200
        alert_triggered = "SUPPRESSED"
        alert_severity = "P3 - Low Warning"
        soar_playbook = [
            {"step": 1, "component": "Browser Isolation (RBI)", "action": "Launch remote disposable browser container session", "status": "ACTIVE", "badge": "bg-yellow-900 text-yellow-200"},
            {"step": 2, "component": "SIEM Telemetry Feed", "action": "Tagged transaction with ML risk flag for session audit", "status": "RECORDED", "badge": "bg-gray-800 text-gray-200"}
        ]
    else:
        risk_level = "LOW"
        risk_color = "#38a169"  # Green
        soc_action = "ALLOW TRAFFIC (WHITELISTED)"
        http_status = 200
        alert_triggered = "SUPPRESSED"
        alert_severity = "P4 - Informational"
        soar_playbook = [
            {"step": 1, "component": "Perimeter Gateway", "action": "Allow normal TLS / HTTP traffic pass-through", "status": "PERMITTED", "badge": "bg-green-900 text-green-200"}
        ]

    classification = "Malicious" if risk_score >= 50 else "Legitimate"

    # Keywords found
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()]

    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        domain = parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        domain = 'unknown'

    features_dict = {}
    for name, val in zip(FEATURE_NAMES, features_raw):
        features_dict[name] = {
            'value': val,
            'description': FEATURE_DESCRIPTIONS.get(name, '')
        }

    return {
        'url': url,
        'domain': domain,
        'classification': classification,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'http_status': http_status,
        'alert_triggered': alert_triggered,
        'alert_severity': alert_severity,
        'soc_action': soc_action,
        'soar_playbook': soar_playbook,
        'models': {
            'logistic_regression': {
                'prediction': 'Malicious' if lr_pred == 1 else 'Legitimate',
                'probability_malicious': round(lr_prob * 100, 2),
                'probability_legit': round((1.0 - lr_prob) * 100, 2),
                'accuracy': 98.15
            },
            'svm': {
                'prediction': 'Malicious' if svm_pred == 1 else 'Legitimate',
                'probability_malicious': round(svm_prob * 100, 2),
                'probability_legit': round((1.0 - svm_prob) * 100, 2),
                'accuracy': 98.10
            }
        },
        'features': features_dict,
        'keywords_detected': found_keywords,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def generate_simulated_event(custom_url=None):
    """Generates a realistic real-time Splunk proxy log event."""
    client_ips = [f"10.10.{random.randint(10, 50)}.{random.randint(2, 250)}" for _ in range(10)]
    users = ['j.smith', 'a.patel', 'm.jones', 'd.taylor', 'e.brown', 'c.wilson', 'svc_scanner', 'b.sharma', 'r.davis']

    if custom_url:
        url = custom_url
    else:
        sample_urls = [p['url'] for p in PRESET_DEMO_URLS] + [
            "https://stackoverflow.com/search?q=machine+learning",
            "http://netflix-account-verify.ru/login",
            "https://en.wikipedia.org/wiki/Support_vector_machine",
            "http://paypal-security-alert-center.xyz/auth",
            "https://docs.python.org/3/library/urllib.parse.html",
            "http://104.244.42.1/admin/login.php",
            "https://www.nytimes.com/section/technology"
        ]
        url = random.choice(sample_urls)

    analysis = analyze_url(url)
    c_ip = random.choice(client_ips)
    user = random.choice(users)

    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'client_ip': c_ip,
        'user': user,
        'dest_domain': analysis['domain'],
        'url': url,
        'http_status': analysis['http_status'],
        'ml_prediction': analysis['classification'],
        'risk_score': analysis['risk_score'],
        'risk_level': analysis['risk_level'],
        'alert_triggered': analysis['alert_triggered'],
        'soc_action': analysis['soc_action']
    }

    raw_splunk_log = (
        f"{datetime.now().strftime('%b %d %H:%M:%S')} proxy-fw01 web_filter: "
        f'src_ip="{c_ip}" user="{user}" method="GET" url="{url}" domain="{analysis["domain"]}" '
        f'status="{analysis["http_status"]}" ml_classifier="LR_SVM_Ensemble" prediction="{analysis["classification"]}" '
        f'risk_score={analysis["risk_score"]} risk_level="{analysis["risk_level"]}" '
        f'alert="{analysis["alert_triggered"]}" action="{analysis["soc_action"]}"'
    )

    return log_entry, raw_splunk_log
