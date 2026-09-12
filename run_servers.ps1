Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Starting Fair Student-Support Prioritization System" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$env:Path = "$env:LOCALAPPDATA\Programs\nodejs;$env:Path"

Write-Host "Starting Backend API (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2

Write-Host "Starting Frontend UI (Port 5173)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:Path = '$env:LOCALAPPDATA\Programs\nodejs;' + `$env:Path; cd frontend; npm run dev -- --host 127.0.0.1 --port 5173"

Write-Host "`nBoth services launched!" -ForegroundColor Cyan
Write-Host "Backend API Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "Frontend Web App: http://127.0.0.1:5173" -ForegroundColor White
