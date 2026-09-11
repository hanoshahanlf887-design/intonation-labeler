@echo off
title Audio Annotation Tool

cd /d "%~dp0"

echo ============================================
echo   Audio Intonation Annotation Tool
echo   Starting up...
echo ============================================
echo.

SET PYTHON_EXE=python

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Python not found. Please install Python 3.9+
    echo         https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Launching web annotation tool...
echo [INFO] Browser will open at http://localhost:8501
echo [INFO] Close this window to stop the server
echo [INFO] If dependencies are missing, install them with:
echo        pip install -r requirements.txt
echo.
echo ============================================

"%PYTHON_EXE%" -m streamlit run app/streamlit_app.py --browser.gatherUsageStats false --server.port 8501

echo.
echo [INFO] Server stopped. Check output above for any errors.
pause
