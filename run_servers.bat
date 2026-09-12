@echo off
echo ===================================================
echo   Starting Fair Student-Support Prioritization System
echo ===================================================

set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"

echo Starting Backend API (Port 8000)...
start "Student Support Backend (FastAPI)" cmd /k "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo Starting Frontend UI (Port 5173)...
start "Student Support Frontend (Vite)" cmd /k "cd frontend && npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Both services are launching!
echo Backend Docs:  http://127.0.0.1:8000/docs
echo Web App UI:    http://127.0.0.1:5173
echo ===================================================
