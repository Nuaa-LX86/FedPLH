# FedMPE

This repository contains the code and evidence package for the ACM TECS manuscript:

**FedMPE: Coordinated Mixed Precision Execution for Federated 3D Medical Image Segmentation**

The repository name remains `FedPLH` so existing links continue to work. It
contains source code, frozen experiment protocols, validated aggregate
evidence, sanitized profile values, and postprocessed summaries. Raw BraTS
data, medical images, original synthesis reports, foundry-bound artifacts, and
model checkpoints are not included.

## Artifact Map

| Manuscript wording | Repository entry point | What it contains |
| --- | --- | --- |
| Source code | `main_experiment.py`, `training/`, `models/`, `simulator/`, `scripts/`, `visualization/`, `tests/` | Training, aggregate gradient clipping and noise, precision allocation, aggregation simulation, postprocessing, plotting, and integrity checks. |
| Frozen experiment protocols | `frozen_experiment_protocols/` | Human-readable protocol entry points for the frozen BraTS partition, client schedules, seeds, and baseline adaptation scope. |
| Validated aggregate evidence | `validated_aggregate_evidence/` | Aggregate-only evidence summaries, hash manifests, integrity reports, and the frozen-trace BEU credit-factor sensitivity result. |
| Sanitized profile values | `sanitized_profile_values/` | Released hardware profile values and the source/derivation record for the manuscript hardware-comparison table. |
| Postprocessed summaries | `postprocessed_summaries/` | Paper-facing JSON/CSV/TEX summaries reconstructed from the frozen runs and hardware profile. |

## Setup

The validated software environment uses Python 3.10.

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

BraTS data must be obtained through its official access process. After download:

```text
python scripts/preprocess_brats_3d.py --dataset_root <BraTS2021> --output_root dataset/processed
```

The five-seed training runs are computationally expensive. The released aggregate evidence can be checked without the BraTS data.

## License and Citation

Original FedMPE code, scripts, documentation, and released aggregate evidence are provided under the MIT License. Third-party datasets, BraTS medical images, external baseline code, model checkpoints, proprietary synthesis reports, standard-cell libraries, and foundry-library-bound artifacts remain governed by their own terms.

If you use this repository, cite the manuscript using the metadata in `CITATION.cff`.

## Verify the Release

Run the release checks from the repository root:

```powershell
python -m pytest -q
python scripts/verify_public_release.py
```

The final paper-facing result summary is:

```text
postprocessed_summaries/paper_results.json
```

The five-policy ablation is released as 25 policy/seed records:

```text
validated_aggregate_evidence/precision_policy_ablation.json
validated_aggregate_evidence/precision_policy_ablation.csv
```

To reproduce the participant-mean BEU credit-factor sensitivity artifact:

```powershell
python plot_beu_boundary.py `
  --profile hardware_profile.json `
  --paper_results postprocessed_summaries/paper_results.json `
  --method HMPE-ACF `
  --history_glob "validated_aggregate_evidence/beu_credit_factor_inputs/seed*/training_history.json" `
  --credit_output validated_aggregate_evidence/beu_credit_factor_sensitivity.reproduced.json `
  --output Fig6_BEU_Boundary_reproduced.pdf
```

The command must find five seeds and 400 round records. The released inputs contain only the three aggregate arrays used by this analysis. The output is an aggregate round-level sensitivity check, not a per-client or straggler bound.

Figures 4 and 5 can be regenerated from the released aggregate summary and
minimal five-seed learning trajectories:

```powershell
python scripts/generate_tpds_figures.py `
  --paper-results postprocessed_summaries/paper_results.json `
  --trajectory-dir postprocessed_summaries/convergence_inputs `
  --output-dir reproduced_figures `
  --manifest reproduced_figures/manifest.json
```

## Evidence Boundaries

- The aggregate gradient accountant output is nominal and is not a formal differential-privacy guarantee.
- Reported energy metrics are modeled local-training compute/memory energy; they exclude clipping-and-noise auxiliary energy, BEU logic, communication, and server aggregation.
- Hardware evidence combines RTL functional verification, separate client/server VCU128-targeted OOC post-route profiles, SAIF power analysis, and trace-based evaluation; it is not a board deployment or complete edge-system measurement.
- The repository provides aggregate evidence and protocol manifests, not raw medical data or private patient-level artifacts.
