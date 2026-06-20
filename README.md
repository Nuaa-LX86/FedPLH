# FedPLH

Official research implementation for:

> **FedPLH: A Cross-Layer Hardware--Software Framework for Budget-Covered Privacy Execution in Edge Federated Medical Learning**

FedPLH combines mixed-precision local training, budget-covered SoftDP operator
scheduling, streaming aggregation, and client-level precision feedback for
federated 3D medical image segmentation.

## Scope and evidence boundaries

This repository reproduces the software and analytical-model evaluation used by
the manuscript. The reported system results combine PyTorch training traces,
RTL-derived parameters, and synthesis-level hardware profiles.

- SoftDP uses aggregate-gradient clipping and noise injection. It is not
  per-sample DP-SGD.
- The reported epsilon is a nominal RDP-accountant output, not a formal
  patient-level differential-privacy guarantee.
- Energy includes modeled local-training compute and memory energy only.
- Communication, server, SoftDP, and BEU auxiliary-logic energy are not modeled.
- Latency is trace-based and profile-guided; it is not measured silicon latency.
- The evaluated workload is BraTS 2021 with 3D U-Net under the frozen protocol.

## Repository structure

```text
artifacts/                 Validated aggregate evidence from 76 frozen runs
dataset/                   Dataset loader and federated partitioning
experiment_protocols/      Frozen splits, partitions, hashes, and run protocol
experiments/               Losses and hardware-model experiments
models/                    3D U-Net, optional SwinUNETR, precision emulator
scripts/                   Preprocessing, execution, and evidence builders
simulator/                 Hardware and aggregation analytical models
tests/                     Integrity and federated-semantics tests
training/                  Federated trainer, aggregation, SoftDP, and ACF
utils/                     Reproducibility and analysis utilities
visualization/             Plot-generation utilities
hardware_profile.json      RTL/synthesis-derived evaluation profile
main_experiment.py         Main experiment entry point
```

## Environment

The validated environment used Python 3.10 and the package versions listed in
`requirements.txt`.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU or CUDA environment if the
default wheel selected by pip is unsuitable.

## Verification

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The release was verified with 20 passing tests.

## Synthetic smoke test

This command checks the training path without downloading BraTS:

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

Synthetic results are functional checks only and must not be used as manuscript
evidence.

## BraTS 2021 data

BraTS data are not redistributed by this repository. Obtain the dataset through
its official access process and comply with its terms. Preprocess it with:

```bash
python scripts/preprocess_brats_3d.py \
  --dataset_root /path/to/BraTS2021 \
  --output_root dataset/processed
```

The expected processed layout is:

```text
dataset/processed/
└── train/
    ├── images/
    │   └── <case-id>.npy
    └── masks/
        └── <case-id>.npy
```

## Frozen 76-run protocol

The exact split definition, partition hashes, methods, seeds, and claim
boundaries are documented in
`experiment_protocols/tetc_semantic_20260615/EXPERIMENT_PROTOCOL.md`.

On Windows PowerShell, the full run can be started with:

```powershell
.\scripts\run_tetc_semantic_final.ps1 -PythonExe python
```

The complete run is computationally expensive. The already validated aggregate
evidence is provided in `artifacts/semantic_evidence_summary.json`.

## Citation

Citation metadata will be added after the manuscript has a stable public
bibliographic record. Until then, cite the manuscript title and this repository
URL.

## License

A software license has not yet been selected. Public visibility alone does not
grant reuse rights. A license should be added before announcing this repository
as an open-source release.
