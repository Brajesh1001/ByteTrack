# PowerShell script to run MOT15 test set evaluation with nano model
# Usage: .\run_mot15_test_nano.ps1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ByteTrack MOT15 Test Set - Nano Model" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Error: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Warning: No virtual environment detected" -ForegroundColor Yellow
    Write-Host "Consider activating virtual environment first" -ForegroundColor Yellow
    Write-Host "Example: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
}

# Display configuration
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Model: Nano (bytetrack_nano_mot17.pth.tar)" -ForegroundColor White
Write-Host "  Dataset: MOT15 Test Set (11 sequences)" -ForegroundColor White
Write-Host "  Device: CPU (use --device gpu for GPU)" -ForegroundColor White
Write-Host ""

# Ask for confirmation
$response = Read-Host "Start testing? (Y/N)"
if ($response -ne "Y" -and $response -ne "y") {
    Write-Host "Testing cancelled." -ForegroundColor Yellow
    exit 0
}

# Run the test script
Write-Host "`nStarting test...`n" -ForegroundColor Cyan

python test_mot15_nano.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTesting completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`nTesting failed with errors." -ForegroundColor Red
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
