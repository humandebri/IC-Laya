# Documentation

詳細profile・反復測定・個別queryの記録・取得JSON・試行ソースはローカル生成物としてGitから除外する。下記文書の「ローカル生成物」は測定時の保存先を示し、checkoutには含まれない。固定入力、コンパクトな参照logits、最終集計は`.gitignore`の明示的な許可リストで管理する。再測定手順と保存対象の方針は[CONTRIBUTING.md](../CONTRIBUTING.md)を参照。

Start with the English [local inference guide](GETTING_STARTED.md). The repository [README](../README.md) summarizes current capabilities and measured limits.

## Current implementation and measurements

The detailed research notes below are primarily in Japanese. Follow their dates, inputs, and Wasm/model hashes when comparing results.

- [INT8 implementation and checkpoint comparisons](INT8.md)
- [Latest optimization measurements](INT8_OPTIMIZATION_V4.md)
- [Rejected optimization trials](INT8_V4_REJECTED_TRIALS.md)
- [Short-query boundary](INT8_SHORT_QUERY.md)
- [Instruction-budget routing](INT8_INSTRUCTION_BUDGET.md)
- [Client-held split-query inference](CLIENT_HELD_QUERY_INFERENCE.md)
- [INT8 conversion costs and input-copy reduction](INT8_CONVERSION_COST.md)
- [Experimental INT8 activation queries](INT8_ACTIVATION_QUERIES.md)
- [Small practical classification probe](INT8_PRACTICAL_128.md)

## Historical material

`MODEL_PORT*`, `IMPLEMENTATION_*`, `HANDOFF.md`, `PERFORMANCE_MEASUREMENTS.md`, `SOURCES.md`, `reference/`, `design-adrs/`, and `design-v2/` preserve earlier investigations and design decisions. Their statements about unimplemented features, missing tests, or performance describe those earlier revisions. They are not the current installation instructions or a current release checklist.

The earlier INT8 optimization reports and their JSON artifacts remain available as measurement history. Keeping a failed trial does not mean its code was adopted.

- [INT8融合処理と判定変更の分析](INT8_FUSION_AND_ERROR_ANALYSIS.md)

- [INT8 query推論の命令数削減探索](INT8_QUERY_OPTIMIZATION_SEARCH.md)

- [queryの通信・分割条件の比較](QUERY_COMMUNICATION_CONDITIONS.md)
- [queryの自動分割と可逆圧縮の実装](QUERY_TRANSPORT_IMPLEMENTATION.md)

- [開始queryと最初の層の融合](QUERY_BEGIN_FUSION.md)
- [BOOM DAO提案のqueryベンチマーク再測定](BOOMDAO_QUERY_BENCHMARK.md)
