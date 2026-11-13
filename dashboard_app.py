"""
=============================================================================
MLCS CA3 Activity 2: Splunk Enterprise Cyber SOC Web Dashboard (Flask Server)
Interactive Demonstration Platform for URL Threat Classification
Models: Logistic Regression vs. Support Vector Machine (Linear SVC)
=============================================================================
"""

import os
import sys
import webbrowser
import threading
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

import act2_core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=STATIC_DIR
)
CORS(app)

# Pre-warm model cache
print("[+] Initializing Machine Learning Engine & Artifacts...")
act2_core.load_models(BASE_DIR)
print("[+] Machine Learning Models (Logistic Regression & SVM) Loaded Successfully.")

# In-memory events store seeded from CSV
EVENTS_CSV = os.path.join(BASE_DIR, "splunk_url_threat_events.csv")
if os.path.exists(EVENTS_CSV):
    events_df = pd.read_csv(EVENTS_CSV)
    GLOBAL_EVENTS = events_df.to_dict(orient='records')
else:
    GLOBAL_EVENTS = []


@app.route('/')
def index():
    """Renders the main Cyber SOC Dashboard."""
    return render_template('dashboard.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Serves static assets from static/ directory or falls back to workspace root
    (for visual artifact PNG images).
    """
    local_static_path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(local_static_path):
        return send_from_directory(STATIC_DIR, filename)
    root_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(root_path):
        return send_from_directory(BASE_DIR, filename)
    return "File not found", 404


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Analyzes a target URL string.
    Returns: classification, risk score, probabilities, 15 features, and SOAR actions.
    """
    try:
        data = request.get_json(force=True)
        url = data.get('url', '').strip()
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        result = act2_core.analyze_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/events', methods=['GET'])
def api_events():
    """Returns the list of proxy events."""
    return jsonify(GLOBAL_EVENTS[:50])


@app.route('/api/simulate-event', methods=['POST'])
def api_simulate_event():
    """Simulates a live inbound web proxy transaction."""
    try:
        data = request.get_json(silent=True) or {}
        custom_url = data.get('url')
        new_event, raw_log = act2_core.generate_simulated_event(custom_url)
        GLOBAL_EVENTS.insert(0, new_event)
        return jsonify(new_event)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-scan', methods=['POST'])
def api_batch_scan():
    """Batch processes a list of URL strings."""
    try:
        data = request.get_json(force=True)
        urls = data.get('urls', [])
        if not urls:
            return jsonify({'error': 'No URLs provided'}), 400

        results = []
        for u in urls:
            u_clean = u.strip()
            if not u_clean:
                continue
            res = act2_core.analyze_url(u_clean)
            results.append({
                'URL': u_clean,
                'Prediction': res['classification'],
                'Risk Score': res['risk_score'],
                'Risk Level': res['risk_level'],
                'LR Conf': f"{res['models']['logistic_regression']['probability_malicious']}%",
                'SVM Conf': f"{res['models']['svm']['probability_malicious']}%",
                'SOC Action': res['soc_action']
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def open_browser(port):
    """Helper to open the default web browser after server starts."""
    webbrowser.open_new(f"http://127.0.0.1:{port}")


if __name__ == '__main__':
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    print("=================================================================")
    print("  MLCS CA3 Activity 2: Splunk Cyber SOC Demonstration Dashboard")
    print(f"  Live Server URL: http://127.0.0.1:{port}")
    print("=================================================================\n")

    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: open_browser(port)).start()

    app.run(host='0.0.0.0', port=port, debug=False)
