# Frozen Experiment Protocols

This directory is the paper-term entry point for **frozen experiment protocols**.

The final matched protocols are:

```text
frozen_experiment_protocols/tpds_operand_complete_20260902/run_manifest.json
frozen_experiment_protocols/tpds_sota_adapters_20260902/run_manifest.json
```

Associated protocol artifacts:

| File or directory | Role |
| --- | --- |
| `frozen_experiment_protocols/tpds_operand_complete_20260902/run_manifest.json` | Seven matched execution paths, five seeds, and operand-complete quantization settings. |
| `frozen_experiment_protocols/tpds_sota_adapters_20260902/run_manifest.json` | FedEvi and FedCLAM adapter protocol. |
| `frozen_experiment_protocols/shared_brats_partition_alpha0p5/` | Immutable partition evidence and aggregate client distribution reused by both TPDS matched runs. |

Boundary: the protocols define seeds, partitions, schedules, and evaluation scope. They do not include raw BraTS data.
