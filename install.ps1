# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
pip install -e .[dev]
if ($LASTEXITCODE -ne 0) {
    Write-Host "Retrying without dev dependencies..." -ForegroundColor Yellow
    pip install -e .
}
