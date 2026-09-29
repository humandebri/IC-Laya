# 一次資料と確認範囲

> Historical note: this document describes an earlier revision. Statements about missing implementations, tests, or measurements are not the current project status. Start with the [English local inference guide](GETTING_STARTED.md) and [current measurement index](README.md).

確認日: 2026-09-19（JST）。以下のURLは確認した公開資料。main/stableは変わり得るため、実装時はcheckpoint、ソース、toolchainのrevisionとhashを固定する。今回はweightsの取得・実モデル推論・Rust/Wasm build・ICP deployは実施していない。

| ID | 資料 | 確認範囲 |
|---|---|---|
| S01 | [Laya model card](https://huggingface.co/convaiinnovations/laya) | 三primitive、option-marker architecture、モデル系列。ベンチ速度をICP性能へ転用しない。 |
| S02 | [Laya typed-decisions model card](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 特化対象の限界、temperature_by_optionsの優先による校正注意点。本番taskの精度保証ではない。 |
| S03 | [Laya rl_common.py](https://huggingface.co/convaiinnovations/laya/blob/main/rl_common.py) | rendering、marker、head、act branch、entropy由来診断。元rendererの無言切詰めをそのまま配備しない。 |
| S04 | [Laya rl_agent_api.py](https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_api.py) | 質問ごとの入力作成、temperature解決、Scoreの期待段階、Noulのp[1]。 |
| S05 | [Candle modernbert.rs](https://raw.githubusercontent.com/huggingface/candle/main/candle-transformers/src/models/modernbert.rs) | backbone、GeGLU、local mask、出力経路。Laya checkpointのICP互換を証明するものではない。 |
| S06 | [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) | activation等の既定値。採用する参照環境のversionと一致させる。 |
| S07 | [ICP Resource limits](https://docs.internetcomputer.org/references/resource-limits/) | update 40B instructions、wasm32 heap 4GiB等。20B/3GiBは本設計の目標であり公式上限ではない。 |
| S08 | [ICP Safe retries and idempotency](https://docs.internetcomputer.org/guides/canister-calls/idempotency/) | nonce、dedup、結果不明、時間窓後の回復と照合。 |
| S09 | [ICP security — Inter-canister calls](https://docs.internetcomputer.org/guides/security/inter-canister-calls/) | awaitをまたぐstateと権限の再検査、障害処理。 |
| S10 | [ICRC-1 specification](https://github.com/dfinity/ICRC-1/blob/main/standards/ICRC-1/README.md) | caller所有accountのdebit、transfer args、dedupのSHOULD要件。ledger固有検証は別途必要。 |
| S11 | [Candid type reference](https://docs.internetcomputer.org/references/candid-spec/) | wireでのrecord/variant/principal等とRust SDKの型復元の境界。 |
| S12 | [Rust visibility and privacy](https://doc.rust-lang.org/reference/visibility-and-privacy.html) | module/private fieldの保護。意味的正しさや分散した一回性を保証するものではない。 |

この文書のcrate構成、Compact128、質問数、上限、型名、API名、状態機械、計測目標は設計上の選択であり、上記資料が公式に推奨した値ではない。

## v0.1 implementation references

- Rust CDK Call API: https://docs.rs/ic-cdk/0.20.3/ic_cdk/call/struct.Call.html
- Stable memory API: https://docs.rs/ic-cdk/0.20.3/ic_cdk/stable/index.html
- Candle Tensor API: https://docs.rs/candle-core/0.11.0/candle_core/struct.Tensor.html
- ModernBERT source: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/modernbert.rs
- Transformers ModernBERT source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py

These references informed the source implementation. Their existence is not a successful build or runtime test. Full Laya upstream sources/weights could not be fetched in the authoring environment, so no claim of checkpoint parity is made.
