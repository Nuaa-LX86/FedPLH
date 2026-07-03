# Paper-Facing Artifacts

This directory exposes the artifact names used in the manuscript.

## Validated Aggregate Evidence

`validated_aggregate_evidence.json` contains the aggregate metrics and paired
comparisons produced from the frozen final experiment:

- 40 main-comparison runs;
- 9 ACF mechanism-ablation runs;
- 27 strong-heterogeneity runs;
- 76 validated runs in total.

`semantic_evidence_summary.json` is retained as a backward-compatible alias for
the same validated aggregate evidence.

## Sanitized Profile Values

`sanitized_profile_values.json` mirrors the released `hardware_profile.json`
values used by the trace-based system model. These values are sanitized
profile inputs, not complete original synthesis reports or foundry-bound PPA
artifacts.

## Postprocessed Summaries

`postprocessed_summaries/` contains paper-facing summaries reconstructed from
the frozen run evidence:

- `paper_results.json`
- `paper_results.csv`
- `main_table_rows.tex`
- `representative_history.json`
- `acf_table_rows.tex`

The JSON summary is sanitized for repository release and does not expose local
absolute paths to private run directories.

## Boundaries

Raw BraTS data, patient-level private artifacts, model checkpoints, complete
per-run output directories, complete original synthesis reports, proprietary
standard-cell libraries, and foundry-library-bound artifacts are not
redistributed.

The evidence boundaries in the root README and frozen experiment protocol apply
to every value in this artifact package.
