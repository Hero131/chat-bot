@echo off
title Host Streamlit App Online
cd /d "%~dp0"

echo ====================================================
echo        Starting Local Web Server (Streamlit)
echo ====================================================
:: Start streamlit in a separate minimized console window
start "Streamlit Server" /MIN cmd /c "python -m streamlit run app.py"

echo Streamlit is starting in the background.
echo.
echo ====================================================
echo        Setting up Public Internet Tunnel
echo ====================================================

if not exist cloudflared.exe (
    echo Downloading Cloudflare Tunnel...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
    echo Download complete!
)

echo.
echo ====================================================
echo SUCCESS! Your app is being hosted on the internet.
echo ====================================================
echo Look for the link ending in ".trycloudflare.com" below.
echo You can share that link with anyone!
echo.
echo To STOP hosting, simply close this window and the Streamlit Server window.
echo.

cloudflared.exe tunnel --url http://127.0.0.1:8501

pause
