# FedPLH

This repository accompanies the manuscript:

> **FedPLH: A Cross-Layer Framework for Budget-Covered Privacy-Operator Execution in Edge Federated Medical Learning**

It contains the implementation and paper-facing evidence used for the FedPLH
study. The evaluated workload is federated 3D medical image segmentation on
BraTS 2021 with a 3D U-Net backbone.

## Contents

The main code is organized as follows:

```text
main_experiment.py         experiment entry point
training/                  federated training, SoftDP processing, ACF, aggregation
models/                    3D U-Net and precision wrappers
simulator/                 hardware-profile-based system model
scripts/                   preprocessing, postprocessing, plots, run scripts
tests/                     integrity and semantics checks
experiment_protocols/      frozen partitions, seeds, schedules, protocol notes
artifacts/                 released aggregate evidence and paper summaries
hardware_profile.json      released hardware-profile values used by the model
```

The manuscript-facing artifacts are:

```text
artifacts/validated_aggregate_evidence.json
artifacts/sanitized_profile_values.json
artifacts/postprocessed_summaries/
experiment_protocols/tetc_semantic_20260615/
```

`artifacts/semantic_evidence_summary.json` is kept as an older alias for the
validated aggregate evidence.

## Scope

This is not a raw-data or full-hardware release. We do not redistribute BraTS
scans, private medical images, model checkpoints, complete per-run output
directories, complete original synthesis reports, standard-cell libraries, or
foundry-bound artifacts.

The released hardware values are sanitized profile values used by the
trace-based model. The SoftDP accountant output reported in the paper is
nominal and is not a formal patient-level differential-privacy guarantee.

## Environment

The validated environment used Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If pip selects the wrong PyTorch build, install the CPU or CUDA wheel matching
your machine first.

## Quick Checks

Run the tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

For a small smoke test that does not require BraTS:

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

Synthetic runs are only software checks and should not be cited as paper
evidence.

## BraTS Data

BraTS 2021 must be obtained through the official access process. After
downloading it, preprocess with:

```bash
python scripts/preprocess_brats_3d.py \
  --dataset_root /path/to/BraTS2021 \
  --output_root dataset/processed
```

## Frozen Protocol

The frozen TETC protocol is described in:

```text
experiment_protocols/tetc_semantic_20260615/EXPERIMENT_PROTOCOL.md
```

The full run is computationally expensive. The released aggregate evidence used
for the manuscript is available under `artifacts/`.

## Citation

If you use this repository, cite the manuscript using `CITATION.cff`. The
citation metadata is marked as a preprint artifact until a stable bibliographic
record is available.

## License

Original FedPLH code, scripts, documentation, and released aggregate evidence
are provided under the MIT License. Third-party datasets, BraTS medical images,
external baseline code, model checkpoints, proprietary synthesis reports,
standard-cell libraries, and foundry-bound artifacts remain governed by their
own terms.
