@echo off
title SwadAI - Smart Fridge Vision & AI Kitchen Coach (Streamlit + WebRTC)
color 0B
cls
echo ======================================================================
echo    SwadAI: Smart Fridge Vision & Live AI Kitchen Coach
echo    Tech Stack: Streamlit | WebRTC | MediaPipe | OpenCV | SQLite | gTTS | LLM
echo ======================================================================
echo.
echo Launching Streamlit WebRTC live camera assistant...
echo.
streamlit run fridge_app.py --server.port 8501
pause
