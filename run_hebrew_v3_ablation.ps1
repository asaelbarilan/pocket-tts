param(
    [int]$Steps = 1000,
    [int]$GradientAccumulation = 8,
    [int]$EvalEvery = 250,
    [int]$EvalSamples = 64,
    [string]$Device = "cuda",
    [switch]$SkipScoring
)

# Required controlled ablation: identical base model, data, seed, optimizer schedule,
# and evaluation coverage. Only loss mode/head multiplier changes.

$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot
$Python = Join-Path $RepoDir ".venv\Scripts\python.exe"
$Artifacts = Join-Path $RepoDir "artifacts\hebrew_v2_8s"
$PromptArtifacts = Join-Path $RepoDir "artifacts\hebrew_v2_23s"
$EvalSet = Join-Path $RepoDir "runs\evaluation\hebrew-v3-controlled.json"
$EvalOutput = Join-Path $RepoDir "runs\evaluation\hebrew-v3-ablation"
$LogDir = Join-Path $RepoDir "logs"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing $Python." }
foreach ($required in @("train_latents.jsonl", "validation_latents.jsonl", "tokenizer.model")) {
    $path = Join-Path $Artifacts $required
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing $path." }
}
if (-not (Test-Path -LiteralPath $EvalSet)) {
    if (-not (Test-Path -LiteralPath (Join-Path $PromptArtifacts "validation.jsonl"))) {
        throw "Missing $EvalSet and preserved prompt audio under $PromptArtifacts."
    }
    New-Item -ItemType Directory -Force (Split-Path $EvalSet) | Out-Null
    & $Python -m hebrew_training.build_eval_set `
        --artifacts $Artifacts $PromptArtifacts `
        --output $EvalSet `
        --sentences-per-group 8 `
        --speakers-per-group 4 `
        --asr-floor-clips 128
    if ($LASTEXITCODE -ne 0) { throw "Controlled evaluation-set build failed." }
}

$arms = @(
    @{ Name = "ablation-flow-m1-s$Steps"; Loss = "flow"; Multiplier = 1 },
    @{ Name = "ablation-flow-m8-s$Steps"; Loss = "flow"; Multiplier = 8 },
    @{ Name = "ablation-fmlsd-m8-s$Steps"; Loss = "fm-lsd"; Multiplier = 8 }
)
$freeGb = [math]::Round(([System.IO.DriveInfo]::new($RepoDir.Substring(0, 1))).AvailableFreeSpace / 1GB, 1)
if ($freeGb -lt 8) {
    throw "Only $freeGb GB free; the three final checkpoints need about 3 GB plus headroom."
}
foreach ($arm in $arms) {
    $runDir = Join-Path $RepoDir ("runs\{0}" -f $arm.Name)
    if (Test-Path -LiteralPath $runDir) { throw "Run already exists: $runDir" }
}

if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }
$checkpoints = @()
Push-Location $RepoDir
try {
    & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA check failed." }
    foreach ($arm in $arms) {
        $runDir = Join-Path $RepoDir ("runs\{0}" -f $arm.Name)
        $logFile = Join-Path $LogDir ("{0}.log" -f $arm.Name)
        Write-Host "Starting $($arm.Name) -> $runDir"
        $trainArgs = @(
            "-m", "hebrew_training.train",
            "--train-manifest", (Join-Path $Artifacts "train_latents.jsonl"),
            "--validation-manifest", (Join-Path $Artifacts "validation_latents.jsonl"),
            "--tokenizer", (Join-Path $Artifacts "tokenizer.model"),
            "--run-dir", $runDir,
            "--base-language", "english",
            "--loss-mode", $arm.Loss,
            "--head-batch-multiplier", $arm.Multiplier,
            "--steps", $Steps,
            "--gradient-accumulation", $GradientAccumulation,
            "--eval-every", $EvalEvery,
            "--eval-samples", $EvalSamples,
            "--save-every", $Steps,
            "--seed", "1337",
            "--device", $Device
        )
        & $Python @trainArgs 2>&1 | Tee-Object -FilePath $logFile
        if ($LASTEXITCODE -ne 0) { throw "$($arm.Name) failed with exit code $LASTEXITCODE." }
        $checkpoints += Join-Path $runDir ("checkpoint-{0:D7}" -f $Steps)
    }

    if (-not $SkipScoring) {
        Write-Host "Scoring all three arms on the fixed four-group evaluation set."
        $evalArgs = @(
            "-m", "hebrew_training.eval_checkpoints",
            "--eval-set", $EvalSet,
            "--tokenizer", (Join-Path $Artifacts "tokenizer.model"),
            "--checkpoints"
        ) + $checkpoints + @(
            "--output-dir", $EvalOutput,
            "--device", $Device
        )
        & $Python @evalArgs
        if ($LASTEXITCODE -ne 0) { throw "Controlled evaluation failed." }
    }
}
finally {
    Pop-Location
}

Write-Host "Ablation complete. Metrics are in each run directory."
if (-not $SkipScoring) { Write-Host "WER/CER results: $(Join-Path $EvalOutput 'eval_results.json')" }
