# PowerShell script to replicate Makefile functionality on Windows
# Usage: .\make.ps1 install | .\make.ps1 train | .\make.ps1 serve | etc.

param(
    [Parameter(Mandatory=$true)]
    [string]$Target
)

$ErrorActionPreference = "Stop"

switch ($Target) {
    "install" {
        Write-Host "Installing dependencies..." -ForegroundColor Green
        pip install -e .[dev]
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Retrying without dev dependencies..." -ForegroundColor Yellow
            pip install -e .
        }
    }
    "train" {
        Write-Host "Training model..." -ForegroundColor Green
        housing-train --config configs/train.yaml
    }
    "serve" {
        Write-Host "Starting server..." -ForegroundColor Green
        uvicorn housing_model.service:app --host 0.0.0.0 --port 8000
    }
    "test" {
        Write-Host "Running tests..." -ForegroundColor Green
        pytest -q
    }
    "docker-build" {
        Write-Host "Building Docker image..." -ForegroundColor Green
        docker build -t housing-model:latest .
    }
    "docker-run" {
        Write-Host "Running Docker container..." -ForegroundColor Green
        docker run --rm -p 8000:8000 housing-model:latest
    }
    "help" {
        Write-Host "Available commands:" -ForegroundColor Cyan
        Write-Host "  .\make.ps1 install      - Install dependencies"
        Write-Host "  .\make.ps1 train        - Train the model"
        Write-Host "  .\make.ps1 serve        - Start the API server"
        Write-Host "  .\make.ps1 test         - Run tests"
        Write-Host "  .\make.ps1 docker-build - Build Docker image"
        Write-Host "  .\make.ps1 docker-run   - Run Docker container"
        Write-Host "  .\make.ps1 help         - Show this help message"
    }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host "Run '.\make.ps1 help' for available commands" -ForegroundColor Yellow
        exit 1
    }
}
