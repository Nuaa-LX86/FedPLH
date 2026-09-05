# Sanitized Profile Values

This directory is the paper-term entry point for **sanitized profile values**.

The released trace-simulator profile is:

```text
hardware_profile.json
```

The corresponding VCU128 implementation evidence is summarized in:

```text
sanitized_profile_values/fpga_vcu128_integrated_profile.json
sanitized_profile_values/selected_operating_points.json
sanitized_profile_values/rtl_test_manifest.json
sanitized_profile_values/saif_power_manifest.json
```

The provenance record for the manuscript hardware-comparison table is:

```text
sanitized_profile_values/hardware_comparison_provenance.csv
```

It records the cited implementation scope, retained source-reported values, omitted or unverified fields, and the status of values from the earlier comparison table.

It provides the hardware-profile values used by the trace-based system model, including:

- clock frequency and memory bandwidth;
- precision-dependent operation throughput assumptions;
- compute and memory energy coefficients;
- clipping-and-noise operator cost parameters;
- server-aggregator pipeline depth, streaming throughput, and software conversion factor.

Boundary: these are sanitized profile values used by the model. The repository does not include complete original synthesis reports, proprietary standard-cell libraries, foundry-library-bound artifacts, or a complete independently reproducible PPA package.

The selected client and server cores are separate OOC implementations targeting
the same VCU128 device. Their combined footprint is an arithmetic sum, not a
co-deployed top-level design.
