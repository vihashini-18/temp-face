
& "$PSScriptRoot\env\Scripts\activate"

$ErrorActionPreference = 'Stop'
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Face Recognition System" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan




$env:KMP_DUPLICATE_LIB_OK = "TRUE"




function Test-PythonAvailable {
    try {
        $py = & python -c "import sys; print(sys.version)" 2>$null
        if (-not $py) { return $false }
        return $true
    } catch { return $false }
}

if (-not (Test-PythonAvailable)) {
    Write-Host "ERROR: Python not found in this shell. Please activate your environment first:" -ForegroundColor Red
    Write-Host "  conda activate face_reg" -ForegroundColor Yellow
    Pause
    exit 1
}


$modelPath = Join-Path $PSScriptRoot 'models\yolov8n-face.pt'
if (-not (Test-Path $modelPath)) {
    Write-Host "YOLO face model not found. Downloading..." -ForegroundColor Yellow
    $modelUrl = 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt'
    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath -UseBasicParsing
        Write-Host "Model downloaded to: $modelPath" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Could not download YOLO model automatically. Ensure the file exists at: $modelPath" -ForegroundColor Red
    }
}


$faissPath = Join-Path $PSScriptRoot 'embeddings\faiss_index.bin'
$labelsPath = Join-Path $PSScriptRoot 'embeddings\labels.pkl'
if (-not (Test-Path $faissPath) -or -not (Test-Path $labelsPath)) {
    Write-Host "Embeddings not found. Generating now..." -ForegroundColor Yellow
    & python src/precompute_embeddings.py
}


Write-Host "Starting video stream... Press 'q' to exit." -ForegroundColor Green
& python run_video.py

Write-Host "Done." -ForegroundColor Gray
