# FedPLH

This repository accompanies the manuscript:

> **FedPLH: A Cross-Layer Framework for Budget-Covered Privacy-Operator Execution in Edge Federated Medical Learning**

The codebase contains the FedPLH training pipeline, the hardware-profile-driven
system model, and the aggregate evidence used in the paper. The main workload is
federated 3D medical image segmentation on BraTS 2021 with a 3D U-Net backbone.

## What Is Included

```text
main_experiment.py         Main experiment entry point
training/                  Federated training, aggregation, SoftDP, and ACF
models/                    3D U-Net, optional SwinUNETR, precision wrapper
simulator/                 Hardware/profile-based system model
scripts/                   Preprocessing, run scripts, postprocessing, plots
tests/                     Integrity and semantics checks
experiment_protocols/      Frozen partitions, seeds, schedules, and protocol notes
artifacts/                 Released aggregate evidence and paper summaries
hardware_profile.json      Released hardware-profile values used by the model
```

The names under `artifacts/` follow the wording used in the manuscript:

| Artifact | Path |
| --- | --- |
| validated aggregate evidence | `artifacts/validated_aggregate_evidence.json` |
| sanitized profile values | `artifacts/sanitized_profile_values.json` |
| postprocessed summaries | `artifacts/postprocessed_summaries/` |
| frozen experiment protocol | `experiment_protocols/tetc_semantic_20260615/` |

`artifacts/semantic_evidence_summary.json` is kept as an older alias for the
validated aggregate evidence.

## Scope Notes

The repository is not a raw-data or full-hardware release.

- BraTS scans are not redistributed.
- Model checkpoints and complete per-run output directories are not included.
- The SoftDP implementation uses aggregate-gradient clipping and noise
  injection, not per-sample DP-SGD.
- The reported epsilon is a nominal RDP-accountant output, not a formal
  patient-level DP guarantee.
- Reported energy is modeled local-training compute/memory energy. It does not
  include SoftDP auxiliary energy, BEU auxiliary logic, communication, or server
  aggregation energy.
- Hardware numbers are released as sanitized profile values used by the
  trace-based model; complete synthesis reports and foundry-bound artifacts are
  not included.

## Environment

The validated environment used Python 3.10. Install the Python dependencies with:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If pip selects the wrong PyTorch build for your machine, install the matching
CPU or CUDA wheel from the PyTorch instructions first.

## Quick Check

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The release was checked with 20 passing tests in the validated environment.

For a small functional check without BraTS, run:

```bash
python main_experiment.py \
  --step train \
  --allow_synthetic \
  --rounds 1 \
  --clients 10 \
  --client_fraction 0.1 \
  --local_epochs 1 \
  --img_size 16 \
  --batch_size 2 \
  --suite sota \
  --scenarios FP32_noDP \
  --deterministic \
  --output_root outputs/smoke
```

Synthetic runs are only smoke tests and should not be cited as paper evidence.

## BraTS Data

BraTS 2021 data must be obtained through the official access process. After
downloading it, preprocess with:

```bash
python scripts/preprocess_brats_3d.py \
  --dataset_root /path/to/BraTS2021 \
  --output_root dataset/processed
```

Expected processed layout:

```text
dataset/processed/
`-- train/
    |-- images/
    |   `-- <case-id>.npy
    `-- masks/
        `-- <case-id>.npy
```

## Frozen Protocol

The frozen 76-run protocol is described in:

```text
experiment_protocols/tetc_semantic_20260615/EXPERIMENT_PROTOCOL.md
```

On Windows PowerShell, the full run can be started with:

```powershell
.\scripts\run_tetc_semantic_final.ps1 -PythonExe python
```

This run is computationally expensive. The already released aggregate evidence
is in:

```text
artifacts/validated_aggregate_evidence.json
```

## Citation

If you use this repository, cite the manuscript using the metadata in
`CITATION.cff`. The citation record is intentionally marked as a preprint
artifact until a stable public bibliographic record is available.

## License

Original FedPLH code, scripts, documentation, and released aggregate evidence
are provided under the MIT License. Third-party datasets, BraTS medical images,
external baseline code, model checkpoints, proprietary synthesis reports,
standard-cell libraries, and foundry-library-bound artifacts remain governed by
their own terms.
