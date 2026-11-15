@echo off
title Face Recognition Attendance Monitoring System
cd /d "%~dp0"
echo ============================================================
echo      Face Recognition Attendance Monitoring System
echo ============================================================
echo.
echo [1] Launch Desktop GUI (Tkinter + OpenCV)
echo [2] Launch Web Browser GUI (Streamlit)
echo.
set /p opt="Enter choice [1 or 2, default=1]: "
if "%opt%"=="2" (
    echo.
    echo Starting Web GUI in your browser...
    python -m streamlit run app.py
) else (
    echo.
    echo Starting Desktop GUI...
    python main.py
)
