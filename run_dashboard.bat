@echo off
title MLCS CA3 Activity 2 - Splunk Cyber SOC Dashboard
color 0A

echo =========================================================================
echo   MLCS CA3 Activity 2: Legitimate vs. Malicious URL Threat Dashboard
echo   Machine Learning Models: Logistic Regression vs. Linear SVM
echo =========================================================================
echo.
echo [*] Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH!
    pause
    exit /b 1
)

echo [*] Launching Splunk SOC Demonstration Web Dashboard...
echo [*] Dashboard will open automatically in your default browser at:
echo     http://127.0.0.1:5000
echo.
echo [*] Press Ctrl+C in this terminal window to stop the server when finished.
echo =========================================================================
echo.

python dashboard_app.py
pause
