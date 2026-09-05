# Postprocessed Summaries

This directory is the paper-term entry point for **postprocessed summaries**.

Primary paper-facing summaries:

| File | Role |
| --- | --- |
| `tecs_submission_results.json` | Canonical result assembled from matched baselines and the frozen Progress only runs. |
| `convergence_inputs/` | Minimal `round`, `val_dice`, and `train_loss` arrays for the five representative runs. |
| `generated_result_values.tex` | Generated utility and system result macros. |
| `generated_ablation_values.tex` | Generated five-policy ablation macros. |
| `generated_fpga_values.tex` | Generated client and server FPGA profile macros. |

The public package is assembled and checked with:

```text
scripts/assemble_tecs_submission_results.py
scripts/analyze_tecs_precision_policy_ablation.py
scripts/export_tecs_public_artifacts.py
scripts/verify_public_release.py
```
