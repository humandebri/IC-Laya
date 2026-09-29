# Laya int8 canister

For current English setup instructions, see [Run Laya locally](GETTING_STARTED.md). The measurements below include dated implementation history.

最新の実測は [INT8_OPTIMIZATION_V4.md](INT8_OPTIMIZATION_V4.md) を参照。
128-token Choiceは最終Wasmの単一updateで39.248B命令となり、2 updateの分割推論も完走した。旧Choice schemaの最短28-token入力は約8.495B命令でquery上限5Bを超える。現行packのowner専用[raw queryは最大16 tokens](INT8_SHORT_QUERY.md)で、17以上は推論前に拒否する。前段階は
[INT8_OPTIMIZATION_V3.md](INT8_OPTIMIZATION_V3.md)、
[INT8_F32_WRITEBACK.md](INT8_F32_WRITEBACK.md)、
[INT8_OPTIMIZATION_V2.md](INT8_OPTIMIZATION_V2.md)、初回は
[INT8_PERFORMANCE.md](INT8_PERFORMANCE.md)に記録し、以下の初期測定値は履歴として保持する。

実装日: 2026-09-22。切り出し後のこのリポジトリで検証した記録。
対象は `convaiinnovations/laya-typed-decisions` の固定revision
`f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`。

## 方式

- `ic-laya-int8-pack-v1`。2次元重みは対称int8、出力行ごとのF32 scale。
  各tensorは `[row-major i8 bytes][rows個のlittle-endian F32 scale]`。
  `storage: I8Row`、length、offset、SHA-256をmanifestに記録する。
  バイアスとLayerNormの1次元weightはF32。旧F32 packも読み込める。
- 線形層は入力トークンごとに動的量子化し、int8×int8→int32で積和するW8A8。
  出力はF32に戻し、LayerNorm・RoPE・softmax・attention積・活性化を実行する。
- embedding/qtypeは参照行だけ復元。全重みのF32展開はしない。
- Wasm kernelだけ`simd128`を有効にし、signed i8→i16拡張とinteger dotを使う。
  Candle全体のtarget-featureは変更しない。
- packのハッシュはscaleも覆う。非有限・非正scale、長さ・形状の不一致、破損を拒否する。
- 初期実装は45-tokenでも一括updateの40B命令上限に達した。V4では測定した128-token Choice入力が単一updateで成功した。一般の入力では上限超過の可能性があるため、分割経路も利用できる。
  `start_token_inference_batch(input, request_id, max_steps)` →
  `step_token_inference_batch(job, expected_step, max_steps)` の継続推論を使う。
  28 encoder層・final norm・2 head層・scorerの**32ステップ**を、既定では16ずつまとめる。
  現行実装では128 tokensを**開始＋最初の16ステップ、残り16ステップの計2 update**でも完走済み。
  演算は一括推論と共通で、途中結果だけをheapに保持する。
- APIはowner専用。直前要求と開始位置・max_stepsが同じ再送は同じ結果を返し、古いjob/stepや変更されたmax_stepsは拒否する。
  max_stepsは1〜16。残りステップが少なければ最後まで進める。バッチ内でエラーが起きた場合は途中状態をcommitしない。
  従来の`step_token_inference`も利用可能で、CLIでは`--stepped --steps-per-call 1`を指定する。
  `--profile`は層別計測のため常に1ステップずつ実行する。
  `token_inference_status`で進捗・最終logitsを再取得できる。
  開始統合APIは同じrequest_id・入力・max_stepsの再送に最初の応答を返す。同じIDで内容を変えると拒否する。
  CLIはIDを事前表示し、`--request-id HEX`で現jobの開始を再送できる。後続バッチは直前要求だけを再送できる。
  別IDの新規start・model交換は旧jobを無効にする。upgradeではjobを破棄し、モデルをwarmup後に最初から再実行する。
- `measure_phases`もowner専用。通常推論を許可されたcallerだけでは実行できない。
  `evaluate`と`measure_phases`は登録済みschemaのqtypeを現packの対応表と照合する。
  モデル交換で対応表が変わった旧schemaは`BindingMismatch`となるため、schemaのversionを更新して再登録する。
- `infer_tokens`は単一updateに収まる入力向けの一括推論として残る。
  継続APIはraw logitsと命令数を返し、Receipt・校正・executorの資金移動権限は発行しない。
  既存`evaluate`/executor経路への継続推論の接続は未実装であり、実モデルで一括evaluateの成功は主張しない。

実モデルのcanonical weight bytesはF32 **1,684,119,556** → int8 **422,734,612**。
省略する上流act headは既存設計のまま。これをLayaの全出力互換とは呼ばない。

## 上流との比較

`artifacts/laya_int8_parity.json` に、使用パッケージ、上流コードのhash、pack hash、
入力、logitsを記録した。Noul/Choice/Score各1件と128-token Choice 1件で、
F32は最大絶対誤差 **4.89e-6**。int8は **0.141**、argmaxは4/4一致。
これは限定サンプルであり、実務データの精度・安全性・校正の検証ではない。
量子化後のconfidenceを本番利用するには別途holdout評価と再校正が必要。

この比較で、旧変換設定のdecision head GELUを**ReLU**へ、
Choice/Noul/Scoreのqtype対応を **[0,2,1]** へ修正した。
上流`DecisionModel`はPyTorch TransformerEncoderLayerの既定ReLUを使う。
また上流のbyte-level BPEに合わせ、schemaのoptionには先頭空白を付けてtokenizeする。
旧方式で登録したbyte-level BPEのschemaを移行するときは、versionを更新して再登録する。

## 再現

Python参照・pack変換には `requirements-dev.txt`、上流比較には追加で`transformers`が必要。
実行に使ったバージョンはparity artifactに記録されている。
ローカルモデル・上流コードはgitignore対象で、Wasmへの埋め込みはしない。

```bash
# F32 canonical packから変換（既存出力は上書きしない）
.venv/bin/python tools/quantize_pack.py checkpoints/laya-f32 checkpoints/laya-int8
cargo run --release -p laya-candle --bin laya-infer -- \
  checkpoints/laya-int8 artifacts/laya-noul-input.json

IC_LAYA_CANDLE=1 qrun -- bash tools/build_one.sh decision-engine
icp network start -d
# 新規ローカルcanisterへの初回installのみ。試験identityにはローカルcyclesが必要。
.venv/bin/python tools/canister_infer.py --initialize \
  --stepped --pack checkpoints/laya-int8 --input artifacts/laya-noul-input.json
# 同じモデルで別入力
.venv/bin/python tools/canister_infer.py --stepped --input artifacts/laya-choice-128-input.json
# upgrade後はstableに残るpackから復元
.venv/bin/python tools/canister_infer.py --stepped --warmup --input artifacts/laya-noul-input.json
```

アップロードは1 MiBごとのbinary Candidファイルを使い、argv制限を受けない。
`--initialize`は`install`であり、既存canisterをreinstallしない。
大きなモデルでは作成時のcyclesだけでメモリ増加の凍結閾値を満たせない場合がある。
今回のローカル試験はcanisterに5Tのテスト用cyclesを追加した。

上流比較の再実行:

```bash
cargo build --release -p laya-candle --bin laya-infer
.venv/bin/python tools/check_laya_parity.py \
  --source checkpoints/laya-source \
  --upstream .cache/upstream/src/laya-main/laya/common.py \
  --f32 checkpoints/laya-f32 --int8 checkpoints/laya-int8 \
  --output artifacts/laya_int8_parity.json
```

`choice-128`は計測用に反復した文章を上流のmax_len=128で構築した入力。
アプリ側の入力を黙って切り詰める変更はしていない。

## ローカルcanisterの実測

128-token実モデルが32ステップを完走した。最初のSIMD測定は
`artifacts/int8_laya_canister_128.json` に記録している。

- 最大1ステップ: 4,666,081,397 instructions（40Bの1 update上限内）。
- 初期化＋全ステップの推論本体合計: 138,228,627,166 instructions。
- CLI経由の全ステップwall time: 約28.8秒。ローカル環境の値でありmainnet見積もりではない。
- 管理APIの総メモリ: 960,325,439 bytes。stableを含む値で、heap単体やcold peakではない。
- Wasmとnativeのint8 logitsはbit一致せず、128-tokenケースの最大差は0.0814。
  Wasm対上流F32の最大差は0.1384、argmaxは一致した。

`tools/check_canister_inference_protocol.py` で、直前stepの再送、完了後の実行、
古いstep、不正job、不正入力時の状態保持、非ownerアクセスを検査できる。
結果は `artifacts/int8_canister_protocol.json`。

数値差を調べるため、整数dotだけをRustのscalar実装へ置換した一時Wasmでも
同じ37-token Scoreを実行した。SIMD版とscalar版のlogitsは完全一致した
（`int8_laya_canister_score.json` / `int8_laya_scalar_canister_score.json`）。
このケースのnativeとの差は整数SIMD由来ではない。浮動小数点演算を含む
native/Wasm間の差の発生箇所までは特定しておらず、bit一致は保証しない。
一時scalar版は撤去し、最終canisterはSIMD版に戻している。

検証済み: Rust workspace（candle有効）、candle無効のdecision-engine check、
Wasm build、Python 55 tests、上流F32比較、int8 native/実canister推論、継続APIの再送・権限検査。

最終SIMD版での4入力比較（詳細: `artifacts/int8_canister_comparison.json`）:

| 入力 | tokens | 合計instructions | 最大step | 上流F32との最大logit差 |
|---|---:|---:|---:|---:|
| noul | 45 | 47.658B | 1.611B | 0.015447 |
| choice | 38 | 40.221B | 1.359B | 0.087608 |
| score | 37 | 39.156B | 1.323B | 0.078764 |
| choice-128 | 128 | 138.229B | 4.666B | 0.138386 |

4入力すべてで上流F32とargmaxが一致。ローカルcanister `4caro-hl777-77775-aaaba-cai` は
SIMD版・実モデルwarm状態で残してある。操作identityは `ic-laya-int8`。mainnetへの配置はしていない。
