# 実装自己レビュー

> Historical note: this document describes an earlier revision. Statements about missing implementations, tests, or measurements are not the current project status. Start with the [English local inference guide](GETTING_STARTED.md) and [current measurement index](README.md).

独立監査ではない。v0.2ではRust compilerによる確認を行った。以下はsourceと参照計算、および実際のビルドで出た指摘・修正。

## ビルドして初めて判明した不具合（v0.2）

| 指摘 | 修正・扱い |
|---|---|
| `tools/build_one.sh`がmacOSのbash 3.2で全滅する | `features=()`の空配列展開が`set -u`下でunboundになる。`run()`経由の条件分岐へ変更 |
| `build_one.sh`が`CARGO_TARGET_DIR`を無視してartifactを取り違える | `cargo metadata`の`target_directory`を解決してから`cp` |
| `--features candle`のWasmビルドが不可能 | `getrandom 0.3`が`wasm32-unknown-unknown`でbackendを持たない。`.cargo/config.toml`でcustom backendを選択し、`getrandom_ic.rs`で`__getrandom_v03_custom`を実装 |
| toolchainによってnativeビルドが成否する | `candle-core 0.11.0`の`stdarch_neon_f16`は1.93.0で未安定。`rust-toolchain.toml`で1.97.1に固定 |
| カスタムRNGが暗号学的乱数と誤解される | `getrandom_ic.rs`のdocコメントに、`raw_rand`の代用ではなくtensor初期化とhash seed専用であることを明記。鍵・nonce生成に使わない |

## 設計上の指摘（v0.1から継続）

| 指摘 | 修正・扱い |
|---|---|
| Noul/Scoreを単一scalarだけにすると意味が失われる | 分布を保持しScoreの期待段階・平均・tailを別々に実装 |
| 単純なmax scoreを正答confidenceと呼んでしまう | top1/marginはDiagnostics。校正IDと別に表示 |
| 未校正の結果を認可に流せる | workflow Planはcalibration ID必須。engineでmodel/schema/tokenizer/期限を照合 |
| cache hitがmodel差替え・校正期限切れを迂回し得る | active modelと校正の検査をcache lookupの前に移動。Rust回帰テスト追加 |
| 校正期限が推論時だけ有効で、残りworkflow中に切れる | request期限全体をcoverするcalibrationを要求 |
| 候補順/qtype順を間違えても値だけ正常に見える | option順固定、Noul false→true、manifestのprimitive_to_qtypeと登録値を照合 |
| wire上のScore平均・selected IDを信じる | 全派生値を分布から再計算し、一致しないreceiptを拒否 |
| 型付き結果をそのまま送金capabilityにする | AuthorizedTransferは外部DTOでなくprivate内部型。生成Candidに現れないことを確認済み |
| model判断中の失効・evidence変更・二重advance | snapshot/revision照合、in-flight Busy、stale停止 |
| allowを早期終了すると必要なScoreを省略する | 拒否側のみ早期終了。全required signalが揃うまで認可不可 |
| 金額エラー時に予算だけ先に変わる | compute-before-commit、cloneから一括更新。予算不足回帰テスト追加 |
| 不明な送金にTooOld/BadFeeが来たら予約を解放する | ever_unknownフラグ、後続エラーでは解除しない |
| success後に遅延errorが届く | terminal successを単調に維持し、settlementは一度のみ |
| nonceを変えれば同じ業務を再実行できる | owner登録operationを別途Available/Reserved/Consumedで管理 |
| unknownのままupgradeや期限を跨ぐ | reservation/nonceを維持。通常upgradeは未解決送金で停止 |
| 全量snapshot保存が未認証呼出しにも走る | engine/canister入口に事前caller検査。executor管理APIもowner確認を前段へ |
| 各request内容を権限検査前にcloneしてしまう | 所有者照合後にcloneする順へ変更 |
| source loaderが任意pickle/remoteコードを実行する | safetensorsのみ、明示mapping、有限値/shape/hash検査 |
| mockが実モデルと誤認される | source・backend kind・manifest・CLI・READMEにTEST ONLYを明示 |
| 型/DIDを手書きしてcodeと乖離する | Rustから生成するbuild手順。実際に3 canister分のDIDを生成して確認 |
| 無検証のlive endpointを残す | LimitedLiveを常に拒否。mock handshake後のtest ledgerに限定 |

## 継続リスク

- 上流checkpointのtensor/tokenizer/qtypeは名前とshapeが全206件で一致したが、**数値parityは未確認**。QKV行順・RoPE適用位置・sliding windowの意味・prompt形式が違えば、名前が全部合っていてもlogitsは別物になる。
- **ICP上のinstructions/heap/cyclesは未測定。** local replicaで動作したことは、20B instructions / 2.5GiB heapの達成を意味しない。
- カスタムgetrandom backendは暗号学的乱数ではない。現状その用途は無いが、将来secret生成に流用してはならない。
- local統合試験のledgerはテストダブル。実asset・本番fee・ledger upgradeは未検証。
- `temperature`が上流ではprimitive別・候補数別なのに、現行Candidはスカラー1個。潰すと校正の意味が変わるため、型の設計判断が未解決。
- full snapshotは書込増幅があり、認可済み主体による大量要求に対する本番DoS対策は未完了。
- `allow_caller`のquota再登録やmode変更はowner権限。controller/ownerの侵害を防ぐものではない。
- calibration metadataは信頼するownerの申告を受ける。正しく評価したartifactであることをコード単体は証明しない。
- mock識別文字列は暗号学的な監査証明ではない。指定するmock canisterとownerを信頼するローカル開発用のguardである。
- Human review endpoint、量子化、compactionなどの目標仕様との差分はIMPLEMENTATION_STATUSを参照。

## 次のレビュー

`cargo test --workspace`、Candle featureのWasm build、local replica統合試験は通過した。残るのは元checkpointから生成したgolden input/logitsでのparity、**ICP上のinstruction/heap実測**、実ledger接続である。ビルドとlocal動作の成功を、性能や品質の証拠にしない。

