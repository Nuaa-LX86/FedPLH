# FedMPE

Code and supporting evidence for the manuscript **FedMPE: Coordinated Mixed Precision Execution for Federated 3D Medical Image Segmentation**.

FedMPE uses a Heterogeneous Mixed Precision Engine (HMPE) for local 3D training. A Budget Enabling Unit checks auxiliary operator cost against admitted timing credit, and a streaming aggregator handles precision tagged updates at the server. Aggregate gradient clipping and noise are the auxiliary operator evaluated in the paper.

## Repository contents

- `training/`, `models/`, `simulator/`: training and execution models.
- `scripts/`: result assembly, integrity checks, and manuscript exports.
- `frozen_experiment_protocols/`: dataset partition and experiment protocols.
- `validated_aggregate_evidence/`: aggregate results and hash manifests.
- `sanitized_profile_values/`: hardware values released for verification.
- `postprocessed_summaries/`: paper facing result files.

The canonical submission result is `postprocessed_summaries/tecs_submission_results.json`. It maps the completed five seed `Progress only` runs to the representative FedMPE configuration without changing the frozen histories. The corresponding provenance record is `validated_aggregate_evidence/tecs_primary_result_manifest.json`.

## Checks

From the repository root:

```powershell
python -m pytest -q
python scripts/verify_public_release.py
```

The paper figures derived from data can be regenerated with the scripts in `scripts/` and `plot_beu_boundary.py`. The generated manifests record their inputs and SHA-256 hashes.

BraTS images, model checkpoints, complete synthesis reports, foundry libraries, and other third party material are not redistributed here. The reported energy covers computation and memory during local training. The VCU128 evidence consists of RTL verification and separate implementation profiles for training and aggregation, rather than a hospital deployment or a complete system energy measurement.

## Citation and license

Citation metadata are in `CITATION.cff`. Original code and released aggregate evidence are provided under the MIT License; third party material remains subject to its original terms.
