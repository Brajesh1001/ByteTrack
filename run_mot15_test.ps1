# MOT15 Testing Automation Script for Windows PowerShell
# This script automates the entire MOT15 testing process

param(
    [string]$Step = "all",
    [string]$Split = "val_half",
    [string]$Model = "x",
    [string]$Device = "gpu"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ByteTrack MOT15 Testing Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Color coded messages
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "→ $msg" -ForegroundColor Yellow }
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# Check if Python is available
function Test-Python {
    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python found: $pythonVersion"
        return $true
    }
    catch {
        Write-Error "Python not found. Please install Python 3.7+"
        return $false
    }
}

# Fix cython_bbox issue on Windows
function Fix-CythonBbox {
    Write-Info "Checking cython_bbox installation..."
    
    # Test if cython_bbox is working
    $testResult = python -c "from cython_bbox import bbox_overlaps; print('OK')" 2>&1
    
    if ($testResult -match "OK") {
        Write-Success "cython_bbox is working"
        return $true
    }
    
    Write-Info "Installing pure Python cython_bbox replacement..."
    
    # Check if cython_bbox.py exists in project root
    if (-not (Test-Path "cython_bbox.py")) {
        Write-Error "cython_bbox.py not found. Please ensure all files are present."
        return $false
    }
    
    # Get Python site-packages directory
    $sitePackages = python -c "import site; print(site.getsitepackages()[0])" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        $destination = Join-Path $sitePackages "cython_bbox.py"
        Copy-Item "cython_bbox.py" $destination -Force
        
        # Verify installation
        $verifyResult = python -c "from cython_bbox import bbox_overlaps; print('OK')" 2>&1
        
        if ($verifyResult -match "OK") {
            Write-Success "cython_bbox installed successfully"
            return $true
        }
    }
    
    Write-Error "Failed to install cython_bbox"
    return $false
}

# Step 1: Download and setup MOT15 dataset
function Setup-Dataset {
    Write-Step "Step 1: Setting up MOT15 Dataset"
    
    if (Test-Path "datasets\mot15\train") {
        Write-Info "MOT15 dataset already exists. Skipping download."
        $response = Read-Host "Do you want to re-download? (y/n)"
        if ($response -ne "y") {
            Write-Success "Using existing dataset"
            return $true
        }
    }
    
    Write-Info "Downloading and setting up MOT15 dataset (this may take a while)..."
    python setup_mot15.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Dataset setup complete"
        return $true
    }
    else {
        Write-Error "Dataset setup failed"
        return $false
    }
}

# Step 2: Convert annotations to COCO format
function Convert-Annotations {
    Write-Step "Step 2: Converting Annotations to COCO Format"
    
    if (Test-Path "datasets\mot15\annotations\val_half.json") {
        Write-Info "Annotations already exist. Re-converting..."
    }
    
    python tools\convert_mot15_to_coco.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Annotations converted successfully"
        return $true
    }
    else {
        Write-Error "Annotation conversion failed"
        return $false
    }
}

# Step 3: Check if pretrained model exists
function Test-PretrainedModel {
    param([string]$ModelSize)
    
    $modelPath = "pretrained\bytetrack_${ModelSize}_mot17.pth.tar"
    
    if (Test-Path $modelPath) {
        Write-Success "Model found: $modelPath"
        return $true
    }
    else {
        Write-Error "Model not found: $modelPath"
        Write-Info "Please download the model from:"
        Write-Info "https://github.com/ifzhang/ByteTrack#model-zoo"
        Write-Info "Or run: python download_model.py"
        return $false
    }
}

# Step 4: Run tracking
function Run-Tracking {
    param(
        [string]$Split,
        [string]$Model,
        [string]$Device
    )
    
    Write-Step "Step 3: Running ByteTrack on MOT15"
    
    $configFile = "exps\example\mot\yolox_${Model}_mix_det.py"
    $checkpointFile = "pretrained\bytetrack_${Model}_mot17.pth.tar"
    
    Write-Info "Configuration: $configFile"
    Write-Info "Checkpoint: $checkpointFile"
    Write-Info "Split: $Split"
    Write-Info "Device: $Device"
    Write-Host ""
    
    if (-not (Test-PretrainedModel -ModelSize $Model)) {
        return $false
    }
    
    $args = @(
        "tools\track_mot15.py",
        "--split", $Split,
        "-f", $configFile,
        "-c", $checkpointFile,
        "--device", $Device,
        "--fuse"
    )
    
    if ($Split -eq "test") {
        $args += "--test"
    }
    
    Write-Info "Starting tracking..."
    python @args
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Tracking complete!"
        return $true
    }
    else {
        Write-Error "Tracking failed"
        return $false
    }
}

# Step 5: Show results location
function Show-Results {
    Write-Step "Results"
    
    Write-Host ""
    Write-Success "Tracking Results Location:"
    Write-Host "  YOLOX_outputs\yolox_x_mix_det\track_results\"
    Write-Host ""
    Write-Success "Visualization Videos:"
    Write-Host "  YOLOX_outputs\yolox_x_mix_det\track_vis\"
    Write-Host ""
}

# Main execution
function Main {
    Write-Host ""
    
    if (-not (Test-Python)) {
        exit 1
    }
    
    # Fix cython_bbox on Windows
    Write-Host ""
    if (-not (Fix-CythonBbox)) {
        Write-Error "Failed to setup cython_bbox. See WINDOWS_CYTHON_BBOX_FIX.md for manual fix."
        exit 1
    }
    
    Write-Host ""
    Write-Info "Configuration:"
    Write-Host "  Step: $Step"
    Write-Host "  Split: $Split"
    Write-Host "  Model: yolox_$Model"
    Write-Host "  Device: $Device"
    
    $success = $true
    
    switch ($Step) {
        "setup" {
            $success = Setup-Dataset
        }
        "convert" {
            $success = Convert-Annotations
        }
        "track" {
            $success = Run-Tracking -Split $Split -Model $Model -Device $Device
        }
        "all" {
            $success = Setup-Dataset
            if ($success) {
                $success = Convert-Annotations
            }
            if ($success) {
                $success = Run-Tracking -Split $Split -Model $Model -Device $Device
            }
            if ($success) {
                Show-Results
            }
        }
        default {
            Write-Error "Unknown step: $Step"
            Write-Info "Available steps: setup, convert, track, all"
            exit 1
        }
    }
    
    Write-Host ""
    if ($success) {
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Process completed successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    }
    else {
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "Process failed. Check errors above." -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        exit 1
    }
}

# Show usage information
if ($args -contains "-help" -or $args -contains "--help" -or $args -contains "-h") {
    Write-Host "Usage: .\run_mot15_test.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Step <step>     Step to run: all, setup, convert, track (default: all)"
    Write-Host "  -Split <split>   Dataset split: val_half, train_half, train, test (default: val_half)"
    Write-Host "  -Model <model>   Model size: nano, tiny, s, m, l, x (default: x)"
    Write-Host "  -Device <device> Device: gpu, cpu (default: gpu)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  # Run complete pipeline on validation set"
    Write-Host "  .\run_mot15_test.ps1"
    Write-Host ""
    Write-Host "  # Only setup dataset"
    Write-Host "  .\run_mot15_test.ps1 -Step setup"
    Write-Host ""
    Write-Host "  # Track on test set with small model"
    Write-Host "  .\run_mot15_test.ps1 -Step track -Split test -Model s"
    Write-Host ""
    Write-Host "  # Run on CPU"
    Write-Host "  .\run_mot15_test.ps1 -Device cpu"
    exit 0
}

# Run main function
Main
