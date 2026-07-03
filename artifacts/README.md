# Artifacts

This directory contains the released evidence files for the paper. The files are
aggregate or sanitized artifacts only; raw BraTS data, checkpoints, complete
run directories, and foundry-bound hardware files are not included.

## Files

- `validated_aggregate_evidence.json`
  Main aggregate evidence file used for the paper. It summarizes 76 validated
  frozen runs: 40 main-comparison runs, 9 ACF ablation runs, and 27
  strong-heterogeneity runs.

- `semantic_evidence_summary.json`
  Older name for the same aggregate evidence. Kept so existing links do not
  break.

- `sanitized_profile_values.json`
  Copy of the released hardware-profile values used by the trace-based system
  model. This is not a full synthesis-report or PPA-release package.

- `postprocessed_summaries/`
  Paper-facing summaries generated from the frozen evidence, including
  `paper_results.json`, `paper_results.csv`, generated table rows, and a
  representative history file.

The boundary statements in the root README and in
`experiment_protocols/tetc_semantic_20260615/EXPERIMENT_PROTOCOL.md` apply to
these files.
