# Documentation

Start with the English [local inference guide](GETTING_STARTED.md). The repository [README](../README.md) summarizes current capabilities and measured limits.

## Current implementation and measurements

The detailed research notes below are primarily in Japanese. Follow their dates, inputs, and Wasm/model hashes when comparing results.

- [INT8 implementation and checkpoint comparisons](INT8.md)
- [Latest optimization measurements](INT8_OPTIMIZATION_V4.md)
- [Rejected optimization trials](INT8_V4_REJECTED_TRIALS.md)
- [Short-query boundary](INT8_SHORT_QUERY.md)
- [Instruction-budget routing](INT8_INSTRUCTION_BUDGET.md)
- [Small practical classification probe](INT8_PRACTICAL_128.md)

## Historical material

`MODEL_PORT*`, `IMPLEMENTATION_*`, `HANDOFF.md`, `PERFORMANCE_MEASUREMENTS.md`, `SOURCES.md`, `reference/`, `design-adrs/`, and `design-v2/` preserve earlier investigations and design decisions. Their statements about unimplemented features, missing tests, or performance describe those earlier revisions. They are not the current installation instructions or a current release checklist.

The earlier INT8 optimization reports and their JSON artifacts remain available as measurement history. Keeping a failed trial does not mean its code was adopted.
