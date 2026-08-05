param(
    [int]$Steps = 20000,
    [int]$GradientAccumulation = 8,
    [int]$EvalSamples = 64,
    [int]$EvalEvery = 250,
    # 1.0 GB per checkpoint. At 2000 this run writes 10 checkpoints, about 10 GB.
    # Dropping to 1000 doubles that to 20 GB, which still fits but leaves less headroom
    # alongside the preserved 11 GB hebrew-20k run.
    [int]$SaveEvery = 2000,
    [ValidateSet("flow", "fm-lsd")]
    [string]$LossMode = "flow",
    [int]$HeadBatchMultiplier = 1,
    [double]$LsdFraction = 0.25,
    [string]$Device = "cuda",
    [string]$Resume = ""
)

# Launcher for the CORRECTED Hebrew Pocket TTS experiment (v2).
#
# This does NOT touch runs\hebrew-20k or artifacts\hebrew. Those are preserved per
# AGENTS.md. Everything here reads artifacts\hebrew_v2_8s and writes runs\hebrew-v2-8s.
#
# Data is already prepared. This script only trains; it does not regenerate manifests,
# tokenizer, or latents. To rebuild those, see HEBREW_TRAINING.md.

$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot
$Python = Join-Path $RepoDir ".venv\Scripts\python.exe"
$Artifacts = Join-Path $RepoDir "artifacts\hebrew_v2_8s"
$RunDir = Join-Path $RepoDir "runs\hebrew-v2-8s"
$LogDir = Join-Path $RepoDir "logs"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Write-Stage {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python."
}

# Fail before a 6-hour run rather than during it.
foreach ($required in @("train_latents.jsonl", "validation_latents.jsonl", "tokenizer.model")) {
    $path = Join-Path $Artifacts $required
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing $path. The data pipeline has not been run; see CHANGES.md."
    }
}

$freeGb = [math]::Round((Get-PSDrive -Name ($RepoDir.Substring(0,1))).Free / 1GB, 1)
$needGb = [math]::Ceiling($Steps / $SaveEvery) * 1.0
Write-Stage ("Free disk {0} GB; this run will write about {1} GB of checkpoints" -f $freeGb, $needGb)
if ($freeGb -lt ($needGb + 5)) {
    throw "Not enough free disk for $needGb GB of checkpoints plus headroom."
}

Push-Location $RepoDir
try {
    Write-Stage "Checking CUDA"
    & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA check failed." }

    if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }
    $logFile = Join-Path $LogDir "hebrew-v2-8s.log"
    Write-Stage "Training -> $RunDir"
    Write-Stage "Console log -> $logFile"
    Write-Stage "Metrics -> $(Join-Path $RunDir 'metrics.jsonl')"
    Write-Stage "About 1.08 s/step measured, so $Steps steps is roughly $([math]::Round($Steps * 1.08 / 3600, 1)) hours."

    $trainArgs = @(
        "-m", "hebrew_training.train",
        "--train-manifest", (Join-Path $Artifacts "train_latents.jsonl"),
        "--validation-manifest", (Join-Path $Artifacts "validation_latents.jsonl"),
        "--tokenizer", (Join-Path $Artifacts "tokenizer.model"),
        "--run-dir", $RunDir,
        "--steps", $Steps,
        "--gradient-accumulation", $GradientAccumulation,
        "--eval-every", $EvalEvery,
        "--eval-samples", $EvalSamples,
        "--save-every", $SaveEvery,
        "--loss-mode", $LossMode,
        "--head-batch-multiplier", $HeadBatchMultiplier,
        "--lsd-fraction", $LsdFraction,
        "--device", $Device
    )
    if ($Resume) { $trainArgs += @("--resume", $Resume) }

    & $Python @trainArgs 2>&1 | Tee-Object -FilePath $logFile
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }

    Write-Stage "Done. Checkpoints in $RunDir"
}
finally {
    Pop-Location
}
