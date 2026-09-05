# Validated Aggregate Evidence

This directory is the paper-term entry point for **validated aggregate evidence**.

Key files:

| File | Role |
| --- | --- |
| `hmpe_training_contract_audit.json` | Five-seed, operand-complete HMPE training contract and run hashes. |
| `sota_adapter_five_seed_audit.json` | FedEvi and FedCLAM adapter results and mechanism checks. |
| `tecs_primary_result_manifest.json` | Provenance for mapping the frozen Progress only runs to the representative FedMPE result. |
| `tecs_precision_policy_ablation.json` | Five policies across five seeds, including Dice, latency, energy, and BF16 allocation. |
| `tecs_precision_policy_ablation.csv` | Compact per-run rows from the same 25-run ablation. |
| `tecs_precision_policy_ablation_manifest.json` | Frozen inputs and output hashes for the ablation analysis. |
| `tecs_result_macro_manifest.json` | Inputs, hashes, and closure checks for the manuscript result macros. |
| `tecs_figure_manifest.json` | Inputs and hashes for the data-driven manuscript figures. |
| `beu_credit_factor_sensitivity.json` | Sensitivity of the BEU deadline bound over 400 participant-mean round records. |
| `beu_credit_factor_inputs/` | Sanitized inputs containing only rounds, profiled timing slack, and operator cost. |

This evidence is aggregate and postprocessed. The BEU sensitivity is not a per-client minimum, per-update worst case, or measured straggler critical path. The directory does not expose raw BraTS scans, patient-level artifacts, or model checkpoints.
