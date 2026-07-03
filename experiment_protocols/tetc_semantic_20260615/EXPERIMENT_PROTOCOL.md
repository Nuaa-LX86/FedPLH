# Frozen FedPLH Experiment Protocol

This protocol defines the frozen experiment used to produce the aggregate
evidence in `artifacts/validated_aggregate_evidence.json`. Historical or exploratory
outputs must not be mixed with these results.

## Fixed data definition

- Dataset: 853 processed BraTS patient volumes.
- Global split seed: 0.
- Split sizes: 683 train, 85 validation, 85 test.
- Allocation unit: complete patient volume.
- Entropy: foreground labels 1/2/3 over complete 3D masks; background excluded.
- Client aggregation weight:
  `n_k / sum(n_j for j in participating_clients)`.
- Validation and test holdouts are disjointly assigned to simulated clients
  only for personalized FedBN evaluation. Their frozen assignments are stored
  in each partition evidence file.

## Main experiment

- Frozen partition: `main_alpha0p5_p0`.
- Dirichlet alpha: 0.5.
- Quantity-balanced: 34-35 training volumes per client.
- Training seeds: 0, 1, 2, 3, 4.
- Methods: 8 methods x 5 seeds = 40 runs.

## ACF mechanism ablation

- Same frozen main partition and client schedules as the main experiment.
- Training seeds: 0, 1, 2.
- Static FP8, progress-only, and entropy-only.
- Full ACF is reused from the matching main-experiment seeds.
- Total additional runs: 9.

## Strong heterogeneity

- Frozen partitions: `stress_alpha0p01_p0`, `p1`, and `p2`.
- Same global split seed 0 for every partition.
- Dirichlet alpha: 0.01.
- Quantity-balanced: 34-35 training volumes per client.
- Methods: FedBN, progress-only ACF, and full ACF.
- Training seeds: 0, 1, 2 for every partition.
- Total: 3 methods x 3 partitions x 3 training seeds = 27 runs.

## Claim boundaries

- SoftDP is aggregate-gradient clipping/noising, not per-sample DP-SGD.
- Epsilon is a nominal RDP accountant output, not a formal patient-level DP
  guarantee.
- Energy is modeled local-training compute/memory energy only.
- Communication and server energy are not modeled.
- Latency combines PyTorch event counts with RTL/profile-based analytical
  models; it is not measured silicon latency.
- FedPAQ is an adapted baseline: every selected client performs five local
  optimizer updates, quantizes its model increment, and the server uses the
  common sample-weighted reduction primitive.

## Reproduction

From the repository root on Windows PowerShell:

```powershell
.\scripts\run_tetc_semantic_final.ps1 -PythonExe python
```

The script verifies the frozen partition hashes before launching 76 runs and
then builds the aggregate evidence summary. In this repository release, the
paper-facing aggregate summary is exposed as
`artifacts/validated_aggregate_evidence.json`, with
`artifacts/semantic_evidence_summary.json` retained as a backward-compatible
alias. Use `-Resume` only to continue an interrupted run in the same output
directory.

The full run requires preprocessed BraTS 2021 data under
`dataset/processed/`. It is computationally expensive; the repository includes
the already validated aggregate summary for inspection without rerunning.
