param(
    [string]$Python = "D:\LX\miniconda\envs\acl_cf\python.exe",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$runRoot = ".\audited_runs\tecs_precision_policy_ablation_operand_complete_20260904"
$referenceRoot = ".\audited_runs\tpds_operand_complete_five_seed_20260902"
$profile = ".\hardware_profile_vcu128.json"
$paperResults = ".\postprocessed_summaries\tecs_submission_results.json"
$historyGlob = Join-Path $runRoot "unet\ACF_progress_only\seed*\training_history.json"
$historyPaths = Get-ChildItem -Path $historyGlob | Sort-Object FullName
if ($historyPaths.Count -ne 5) {
    throw "Expected five frozen Progress-only histories, found $($historyPaths.Count)"
}
$hashesBefore = @{}
foreach ($path in $historyPaths) {
    $hashesBefore[$path.FullName] = (Get-FileHash -LiteralPath $path.FullName -Algorithm SHA256).Hash
}

& $Python .\scripts\analyze_tecs_precision_policy_ablation.py `
    --run-root $runRoot `
    --reference-root $referenceRoot `
    --expected-rounds 80 `
    --expected-seeds 0,1,2,3,4 `
    --expected-partition-input-sha256 A193F2783C84D38233BD85BF08B442FB75C85D5C2515E2FD4FBC67786C39902F `
    --expected-partition-output-sha256 FE064DA7DA3259B7BF2E3D83ACEA2ED7E75C58A70E8BA2BB2BF40FA381E5F945 `
    --expected-profile-sha256 BD1A3BC2F3EB512AF8718FEC8DC4897963C27DD5AB14727AA309ABD2E51D76CE `
    --phase full `
    --json-output .\validated_aggregate_evidence\tecs_precision_policy_ablation.json `
    --csv-output .\validated_aggregate_evidence\tecs_precision_policy_ablation.csv `
    --latex-output .\TECS_submission\source\generated_ablation_values.tex `
    --manifest-output .\validated_aggregate_evidence\tecs_precision_policy_ablation_manifest.json
if ($LASTEXITCODE -ne 0) { throw "Precision policy audit failed" }

& $Python .\scripts\assemble_tecs_submission_results.py
if ($LASTEXITCODE -ne 0) { throw "Submission result assembly failed" }

& $Python .\plot_beu_boundary.py `
    --profile $profile `
    --paper_results $paperResults `
    --method FedMPE `
    --history_glob $historyGlob `
    --expected_seed_count 5 `
    --expected_round_count 80 `
    --credit_output .\validated_aggregate_evidence\beu_credit_factor_sensitivity.json `
    --output .\TECS_submission\source\figures\Fig6_BEU_Boundary.pdf
if ($LASTEXITCODE -ne 0) { throw "BEU boundary generation failed" }

& $Python .\scripts\generate_tpds_figures.py `
    --results-dir (Join-Path $runRoot "unet") `
    --final-scenario ACF_progress_only `
    --paper-results $paperResults `
    --output-dir .\TECS_submission\source\figures `
    --manifest .\validated_aggregate_evidence\tecs_figure_manifest.json
if ($LASTEXITCODE -ne 0) { throw "TECS figure generation failed" }

& $Python .\scripts\export_tpds_result_values.py `
    --paper-results $paperResults `
    --primary-run-root $runRoot `
    --sota-audit .\validated_aggregate_evidence\sota_adapter_five_seed_audit.json `
    --beu-boundary .\TECS_submission\source\figures\Fig6_BEU_Boundary.json `
    --credit-sensitivity .\validated_aggregate_evidence\beu_credit_factor_sensitivity.json `
    --output .\TECS_submission\source\generated_result_values.tex `
    --manifest .\validated_aggregate_evidence\tecs_result_macro_manifest.json
if ($LASTEXITCODE -ne 0) { throw "TECS macro generation failed" }

foreach ($path in $historyPaths) {
    $after = (Get-FileHash -LiteralPath $path.FullName -Algorithm SHA256).Hash
    if ($after -ne $hashesBefore[$path.FullName]) {
        throw "Frozen history changed during postprocessing: $($path.FullName)"
    }
}

if (-not $SkipBuild) {
    & .\TECS_submission\build.ps1
    if ($LASTEXITCODE -ne 0) { throw "TECS LaTeX build failed" }
    & $Python .\scripts\qa_tecs_submission.py --final
    if ($LASTEXITCODE -ne 0) { throw "TECS final QA failed" }
}

Write-Host "TECS postprocessing completed without running a training command."
