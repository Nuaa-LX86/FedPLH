param(
    [string]$Python = "python",
    [string]$HardwareProfile = ".\hardware_profile.json",
    [string]$OutputRoot = ".\audited_runs\tpds_final_profile_smoke_20260902"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $HardwareProfile)) {
    throw "Audited VCU128 compatibility profile not found: $HardwareProfile"
}

& $Python .\main_experiment.py `
    --step train `
    --model unet `
    --rounds 2 `
    --clients 20 `
    --data_root .\dataset\processed `
    --output_root $OutputRoot `
    --hardware_profile $HardwareProfile `
    --overwrite `
    --deterministic `
    --val_ratio 0.1 `
    --test_ratio 0.1 `
    --split_seed 0 `
    --partition_seed 0 `
    --partition_file .\frozen_experiment_protocols\shared_brats_partition_alpha0p5\partition_evidence.json `
    --alpha 0.5 `
    --entropy_scope full_volume_foreground `
    --min_client_samples 10 `
    --balance_client_sizes `
    --partition_basis foreground_composition_quantiles `
    --composition_bins 10 `
    --client_fraction 0.2 `
    --local_epochs 2 `
    --img_size 64 `
    --batch_size 2 `
    --lr 0.0001 `
    --hmpe_operand_model quantized_operands `
    --noise_multiplier 0.1 `
    --clip_norm 1.0 `
    --delta 0.00001 `
    --dp_cost_model paper `
    --seed 0 `
    --seeds 0 `
    --suite sota `
    --scenarios FP32_noDP,FP32_softDP,FedBN,Mao_etal,BitFusion,HMPE-ACF_noDP,HMPE-ACF

if ($LASTEXITCODE -ne 0) {
    throw "TPDS final-profile smoke run failed with exit code $LASTEXITCODE"
}

$manifest = Get-Content -LiteralPath (Join-Path $OutputRoot "run_manifest.json") -Raw | ConvertFrom-Json
if ($manifest.status -ne "completed") {
    throw "TPDS final-profile smoke manifest is not completed"
}

Write-Host "TPDS final-profile smoke gate passed: $OutputRoot"
