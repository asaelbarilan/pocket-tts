param(
    [int]$Steps = 12000,
    [int]$GradientAccumulation = 8,
    [int]$EvalSamples = 64,
    [int]$EvalEvery = 250,
    [int]$SaveEvery = 3000,
    [string]$RunName = "hebrew-v3-8s-fmlsd-m8",
    [string]$Device = "cuda",
    [string]$Resume = "",
    [string]$InitFlowCheckpoint = "",
    [switch]$Smoke
)

# New isolated 6-layer experiment. The default is a clean run from Kyutai's released
# English student with the reconstructed 75/25 FM/LSD objective and head multiplier 8.
# It never writes into the completed hebrew-v2-8s directory.

$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot
$Python = Join-Path $RepoDir ".venv\Scripts\python.exe"
$Artifacts = Join-Path $RepoDir "artifacts\hebrew_v2_8s"
$LogDir = Join-Path $RepoDir "logs"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Write-Stage {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

if ($Resume -and $InitFlowCheckpoint) {
    throw "-Resume and -InitFlowCheckpoint are mutually exclusive."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. See HEBREW_TRAINING.md."
}
foreach ($required in @("train_latents.jsonl", "validation_latents.jsonl", "tokenizer.model")) {
    $path = Join-Path $Artifacts $required
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing $path." }
}

$SkipFinalCheckpoint = $false
if ($Smoke) {
    $Steps = 30
    $GradientAccumulation = 1
    $EvalSamples = 8
    $EvalEvery = 10
    $SaveEvery = 0
    $RunName = "hebrew-v3-8s-fmlsd-m8-smoke-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $Resume = ""
    $InitFlowCheckpoint = ""
    $SkipFinalCheckpoint = $true
}

$RunDir = Join-Path $RepoDir ("runs\{0}" -f $RunName)
$LogFile = Join-Path $LogDir ("{0}.log" -f $RunName)
if ((Test-Path -LiteralPath $RunDir) -and -not $Resume) {
    throw "Run directory already exists: $RunDir. Choose a new -RunName."
}
if ($Resume -and -not (Test-Path -LiteralPath $Resume)) {
    throw "Resume checkpoint does not exist: $Resume"
}
if ($InitFlowCheckpoint -and -not (Test-Path -LiteralPath $InitFlowCheckpoint)) {
    throw "Initial flow checkpoint does not exist: $InitFlowCheckpoint"
}

$checkpointCount = if ($SkipFinalCheckpoint) {
    0
} elseif ($SaveEvery -gt 0) {
    [math]::Ceiling($Steps / $SaveEvery)
} else {
    1
}
$freeGb = [math]::Round(([System.IO.DriveInfo]::new($RepoDir.Substring(0, 1))).AvailableFreeSpace / 1GB, 1)
$needGb = $checkpointCount * 1.0
Write-Stage ("Free disk {0} GB; expected checkpoints about {1} GB" -f $freeGb, $needGb)
if (-not $Smoke -and $freeGb -lt ($needGb + 5)) {
    throw "Not enough free disk for $needGb GB of checkpoints plus 5 GB headroom."
}

Push-Location $RepoDir
try {
    & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA check failed." }
    if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }

    $trainArgs = @(
        "-m", "hebrew_training.train",
        "--train-manifest", (Join-Path $Artifacts "train_latents.jsonl"),
        "--validation-manifest", (Join-Path $Artifacts "validation_latents.jsonl"),
        "--tokenizer", (Join-Path $Artifacts "tokenizer.model"),
        "--run-dir", $RunDir,
        "--base-language", "english",
        "--loss-mode", "fm-lsd",
        "--lsd-fraction", "0.25",
        "--head-batch-multiplier", "8",
        "--steps", $Steps,
        "--gradient-accumulation", $GradientAccumulation,
        "--eval-every", $EvalEvery,
        "--eval-samples", $EvalSamples,
        "--save-every", $SaveEvery,
        "--device", $Device
    )
    if ($Resume) { $trainArgs += @("--resume", $Resume) }
    if ($InitFlowCheckpoint) { $trainArgs += @("--init-flow-checkpoint", $InitFlowCheckpoint) }
    if ($SkipFinalCheckpoint) { $trainArgs += "--skip-final-checkpoint" }

    Write-Stage "Run -> $RunDir"
    Write-Stage "Log -> $LogFile"
    Write-Stage "Metrics -> $(Join-Path $RunDir 'metrics.jsonl')"
    & $Python @trainArgs 2>&1 | Tee-Object -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }
    Write-Stage "Completed successfully."
}
finally {
    Pop-Location
}
