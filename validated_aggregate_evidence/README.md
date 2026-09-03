# Validated Aggregate Evidence

This directory is the paper-term entry point for **validated aggregate evidence**.

Key files:

| File | Role |
| --- | --- |
| `hmpe_training_contract_audit.json` | Five-seed, operand-complete HMPE training contract and run hashes. |
| `sota_adapter_five_seed_audit.json` | FedEvi and FedCLAM adapter results and mechanism checks. |
| `tpds_result_macro_manifest.json` | Inputs, hashes, and closure checks for manuscript result macros. |
| `tpds_figure_manifest.json` | Inputs and hashes for data-driven manuscript figures. |
| `tpds_submission_qa.json` | Automated page, citation, placeholder, and LaTeX checks. |
| `validated_aggregate_evidence/beu_credit_factor_sensitivity.json` | Five-seed, 400-round participant-mean sensitivity of BEU coverage to the admitted slack-credit factor. |
| `validated_aggregate_evidence/beu_credit_factor_inputs/` | Sanitized per-seed inputs containing only `round`, `delta_c_cycles`, and `c_priv_cycles` for reproducing the sensitivity artifact. |

Boundary: this evidence is aggregate and postprocessed. The BEU sensitivity is not a per-client minimum, per-update worst case, or measured straggler critical path. The directory does not expose raw BraTS scans, patient-level private artifacts, or model checkpoints.
