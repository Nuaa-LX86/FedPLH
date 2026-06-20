param(
    [string]$PythonExe = "python",
    [string]$OutputRoot = ".\outputs\tetc_semantic_final_20260615",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$protocolRoot = ".\experiment_protocols\tetc_semantic_20260615"
$protocolManifest = Join-Path $protocolRoot "protocol_manifest.json"
$protocol = Get-Content $protocolManifest -Raw | ConvertFrom-Json

function Assert-PartitionHash {
    param(
        [string]$Path,
        [string]$ExpectedHash
    )
    $actualHash = (Get-FileHash $Path -Algorithm SHA256).Hash
    if ($actualHash -ne $ExpectedHash) {
        throw "Partition hash mismatch: $Path"
    }
}

$mainPartition = Join-Path $protocolRoot $protocol.main_partition.path
Assert-PartitionHash `
    -Path $mainPartition `
    -ExpectedHash $protocol.main_partition.sha256
foreach ($partition in $protocol.stress_partitions) {
    Assert-PartitionHash `
        -Path (Join-Path $protocolRoot $partition.path) `
        -ExpectedHash $partition.sha256
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Write-PipelineStatus {
    param(
        [string]$Stage,
        [string]$Status
    )
    @{
        stage = $Stage
        status = $Status
        updated_at = (Get-Date).ToString("o")
        output_root = [System.IO.Path]::GetFullPath($OutputRoot)
    } | ConvertTo-Json | Set-Content `
        -Path (Join-Path $OutputRoot "pipeline_status.json") `
        -Encoding utf8
}

function Invoke-PythonStage {
    param(
        [string]$Stage,
        [string[]]$Arguments
    )
    Write-PipelineStatus -Stage $Stage -Status "running"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-PipelineStatus -Stage $Stage -Status "failed"
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
    Write-PipelineStatus -Stage $Stage -Status "completed"
}

function Get-TrainingArgs {
    param(
        [string]$StageOutput,
        [string]$PartitionFile,
        [int]$PartitionSeed,
        [double]$Alpha,
        [string]$Suite,
        [string]$Scenarios,
        [string]$Seeds
    )
    $arguments = @(
        "main_experiment.py",
        "--step", "train",
        "--model", "unet",
        "--rounds", "80",
        "--clients", "20",
        "--data_root", ".\dataset\processed",
        "--val_ratio", "0.1",
        "--test_ratio", "0.1",
        "--split_seed", "0",
        "--partition_seed", "$PartitionSeed",
        "--partition_file", $PartitionFile,
        "--alpha", "$Alpha",
        "--entropy_scope", "full_volume_foreground",
        "--min_client_samples", "10",
        "--balance_client_sizes",
        "--partition_basis", "foreground_composition_quantiles",
        "--composition_bins", "10",
        "--client_fraction", "0.2",
        "--local_epochs", "2",
        "--img_size", "64",
        "--batch_size", "2",
        "--lr", "1e-4",
        "--noise_multiplier", "0.1",
        "--clip_norm", "1.0",
        "--delta", "1e-5",
        "--dp_cost_model", "paper",
        "--deterministic",
        "--seed", "0",
        "--seeds", $Seeds,
        "--suite", $Suite,
        "--scenarios", $Scenarios,
        "--output_root", $StageOutput
    )
    if ($Resume) {
        $arguments += "--resume"
    }
    return $arguments
}

$coreRoot = Join-Path $OutputRoot "core"
$ablationRoot = Join-Path $OutputRoot "ablation"
$stressRoot = Join-Path $OutputRoot "stress"
$coreScenarios = "FP32_noDP,FP32_softDP,FedBN,FedPAQ,Mao_etal,BitFusion,HMPE-ACF_noDP,HMPE-ACF"

Invoke-PythonStage -Stage "core_40_runs" -Arguments (
    Get-TrainingArgs `
        -StageOutput $coreRoot `
        -PartitionFile $mainPartition `
        -PartitionSeed 0 `
        -Alpha 0.5 `
        -Suite "sota" `
        -Scenarios $coreScenarios `
        -Seeds "0,1,2,3,4"
)

Invoke-PythonStage -Stage "validate_core" -Arguments @(
    "scripts\build_paper_results.py",
    "--results_dir", (Join-Path $coreRoot "unet"),
    "--scenarios", $coreScenarios,
    "--seeds", "0,1,2,3,4",
    "--baseline", "FedBN"
)

Invoke-PythonStage -Stage "acf_ablation_9_runs" -Arguments (
    Get-TrainingArgs `
        -StageOutput $ablationRoot `
        -PartitionFile $mainPartition `
        -PartitionSeed 0 `
        -Alpha 0.5 `
        -Suite "acf_evidence" `
        -Scenarios "ACF_static_FP8,ACF_progress_only,ACF_entropy_only" `
        -Seeds "0,1,2"
)

foreach ($partition in $protocol.stress_partitions) {
    $partitionId = [int]$partition.partition_seed
    $partitionFile = Join-Path $protocolRoot $partition.path
    $partitionOutput = Join-Path $stressRoot "partition$partitionId"
    Invoke-PythonStage -Stage "stress_partition${partitionId}_9_runs" -Arguments (
        Get-TrainingArgs `
            -StageOutput $partitionOutput `
            -PartitionFile $partitionFile `
            -PartitionSeed $partitionId `
            -Alpha 0.01 `
            -Suite "acf_evidence" `
            -Scenarios "ACF_FedBN,ACF_progress_only,ACF_full" `
            -Seeds "0,1,2"
    )
}

Invoke-PythonStage -Stage "validate_all_76_runs" -Arguments @(
    "scripts\build_tetc_semantic_evidence.py",
    "--core_root", $coreRoot,
    "--ablation_root", $ablationRoot,
    "--stress_root", $stressRoot,
    "--protocol_manifest", $protocolManifest,
    "--output", (Join-Path $OutputRoot "semantic_evidence_summary.json"),
    "--rounds", "80"
)

Write-PipelineStatus -Stage "complete" -Status "completed"
