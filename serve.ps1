# Start the API server
Write-Host "Starting server..." -ForegroundColor Green
uvicorn housing_model.service:app --host 0.0.0.0 --port 8000
