# Postprocessed Summaries

This directory is the paper-term entry point for **postprocessed summaries**.

Primary paper-facing summaries:

| File | Role |
| --- | --- |
| `postprocessed_summaries/paper_results.json` | Main five-seed JSON summary used for manuscript tables and result narration. |
| `postprocessed_summaries/paper_results.csv` | Compact CSV view of the paper-facing summary. |
| `postprocessed_summaries/convergence_inputs/` | Minimal `round`, `val_dice`, and `train_loss` arrays for all five seeds. |
| `postprocessed_summaries/generated_result_values.tex` | Generated utility and system-result macros. |
| `postprocessed_summaries/generated_fpga_values.tex` | Generated client/server FPGA profile macros. |

The result and figure exporters are:

```text
scripts/build_paper_results.py
scripts/generate_tpds_figures.py
scripts/export_tpds_result_values.py
```
