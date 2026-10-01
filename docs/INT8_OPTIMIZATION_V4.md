# Laya canister推論の追加改善（V4）

2026-09-25、local IC canisterで実モデルのW8A8推論を測定した。モデルcheckpoint、pack形式、量子化の丸め、F32 scaleの適用順は変更していない。通常の推論APIも維持した。診断用APIはowner専用である。

## 採用結果

| 指標 | 比較用現行版 | 採用版 | 差 |
|---|---:|---:|---:|
| Choice 128 tokens、通常推論32ステップ合計 | 42,841,760,152命令 | 39,274,507,249命令 | **8.327%削減** |
| warmup合計 | 33,913,718,557命令 | 33,913,454,475命令 | 264,082命令削減 |
| warmup中の観測メモリ最大値 | 956,831,731 bytes | 957,030,476 bytes | +198,745 bytes（約0.190 MiB） |
| 最終Wasmサイズ | — | 5,890,316 bytes | — |
| 96入力のlogits最大絶対差 | — | **0.0** | 判定96/96一致 |

上表のWasmはINT8採用版（query診断API追加前）。追加後のWasmも下記で別途再検証した。

128-tokenの現行比較用測定はbaseline-corpus-choice-128.json（ローカル生成物: `artifacts/int8_optimization_v4/baseline-corpus-choice-128.json`）、採用版はfinal-choice-128.json（ローカル生成物: `artifacts/int8_optimization_v4/final-choice-128.json`）。以前のV3記録42,841,802,273命令との差42,121命令はベンチ用APIを追加したWasmの差であり、上表では同じV4比較用Wasmで再測定した値を使用した。pack SHA-256は両方とも`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。

10%削減の目標には届かなかったが、採用条件の1%以上を満たした。96入力の命令数はすべて改善し、削減率の範囲は**8.303〜8.805%**、中央値は**8.709%**。実canister比較の全件データ（ローカル生成物: `artifacts/int8_optimization_v4/final-vs-baseline-canister.json`）に、入力・Wasm・pack・corpusのhash、出力、命令数、判定を記録した。96入力は自然文24件×3 schemaと、35・63・64・65・96・112・127・128 tokensの境界入力×3 schemaで構成する。互換性検査であり、正答率評価ではない。

採用した変更は64行×16列タイルの整数積ループを2回展開したことと、attention QK直前の転置Kに対する不要な`contiguous()`コピーを除いたこと。64×16の選択は主要3形状すべてで勝ち、形状別分岐を増やす根拠はなかった。端数は既存の小タイル経路で処理する。

## 候補の比較

採用しなかった試行の条件、数値、撤回理由、候補ソースへのリンクは[別メモ](INT8_V4_REJECTED_TRIALS.md)にもまとめた。

合成行列積は出力×入力の3形状（3072×1024、5248×1024、1024×2624）を28・38・64・96・128 tokensで測定した。各条件に準備1回・測定3回を行い、JSONに各回、中央値、最小値、最大値、checksumを保存した。表の変化率は15条件の中央値で、負値が改善。合成値のみでは採用を決めていない。

| 候補 | 行列積命令数の中央値変化 | 判定 |
|---|---:|---|
| 元の64×16 | 0% | 比較基準 |
| warmup時にi16へ符号拡張 | +5.120% | 撤回。実128 tokensは44.781B命令。warmupは+5.376B命令、観測メモリは約+446 MiB |
| 出力ブロック別i8配置 | +4.309% | 撤回 |
| 16×8 / 16×16 | +8.770% / +3.488% | 撤回 |
| 32×8 / 32×16 | +6.140% / +0.983% | 撤回 |
| 64×8 | +4.813% | 撤回 |
| 32×16、整数積2回展開 | -7.799% | 64×16展開に劣る |
| **64×16、整数積2回展開** | **-8.893%** | 採用 |
| **最終版64×16、整数積2回展開** | **-8.673%** | 採用Wasmで再測定 |

候補ごとの全15条件とchecksumは[このディレクトリ](../artifacts/int8_optimization_v4/)の`*-matrix.json`に保存した。最終版の15条件すべてで基準より8.328〜8.840%改善し、checksumはすべて一致した。i16配置はwarmup費用も通常推論費用も増えたため、回収可能な推論回数は存在しない。元weightのヒープコピーを解放する試行も含め、採用版には配置変更を残していない。

attentionは個別に比較した。scale・mask・softmaxの連続処理は128-token代表入力で約0.316%改善したが、96入力の比較でlogits最大差0.647、判定2件不一致となり撤回した。QK専用SIMDは採用した通常QK経路より4.20%遅く、最大差0.110で撤回した。AV専用SIMDは16.36%遅く撤回した。`contiguous()`除去と元のsoftmaxの組合せが採用版である。候補の実測JSONとソースの退避は同じartifactディレクトリにある。

診断器は現行の整数積とSIMD書戻し関数を使うよう更新した。採用版の診断（ローカル生成物: `artifacts/int8_optimization_v4/final-components.json`）では整数積が計測器入り行列積の98.65%、98.65%、99.44%を占める。計測器はコード生成を変えるため、この内訳を通常実行の厳密な内訳とはみなさない。通常実行の総命令数は別に保存した。

## 境界と再現

- 最終Wasmをlocal canisterにinstallしてwarmupし、128-token分割推論を2 updateで完走した。分割推論（ローカル生成物: `artifacts/int8_optimization_v4/final-batched-choice-128.json`）と再送・継続・非owner検査（ローカル生成物: `artifacts/int8_optimization_v4/final-protocol.json`）を参照。
- 反復したChoice入力の単一updateは28、96、112、118、120、124、127、**128 tokens**の全点で成功した。128 tokensは39,258,360,558命令。入力内容と全結果は境界測定（ローカル生成物: `artifacts/int8_optimization_v4/final-update-limits.json`）。任意の128-token入力の成功保証ではない。96入力中の127-token Scoreは39,264,165,979命令で単一update成功、128-token境界入力は分割実行で検証した。
- 旧Choice schemaで本文を1 tokenにした28-token入力は**8,494,909,941命令**で、[ICP公式のquery上限5B](https://docs.internetcomputer.org/references/resource-limits/)を超える。後続調査ではschemaを短くして15〜16-tokenのraw入力を作り、owner専用queryで成功した。[条件と品質上の制約](INT8_SHORT_QUERY.md)を参照。通常の`evaluate`はupdateのまま。
- 整数積とF32書戻しのWasm参照テストは920・144・216例でPASS。63・64・65・127・128 tokens、端数列、最大積和、非有限値を含む。`cargo test -p laya-candle --tests`、`cargo test -p decision-engine --features candle`もPASS。

再測定時は同じpackをロードしたlocal canisterに対象Wasmをinstallし、`build/decision-engine.wasm`とcanister module hashを一致させる。主なコマンド:

```bash
python3 tools/benchmark_int8_matrix.py --output artifacts/int8_optimization_v4/final-matrix.json
python3 tools/benchmark_int8_components.py --output artifacts/int8_optimization_v4/final-components.json
python3 tools/measure_warmup.py --output artifacts/int8_optimization_v4/final-warmup.json
python3 tools/make_int8_validation_corpus.py
python3 tools/validate_canister_corpus.py --baseline artifacts/int8_optimization_v4/baseline-canister-corpus.json --output artifacts/int8_optimization_v4/final-vs-baseline-canister.json
python3 tools/measure_update_limits.py --lengths 28,96,112,118,120,124,127,128 --cases choice --output artifacts/int8_optimization_v4/final-update-limits.json
```

全96入力の基準出力は[baseline-canister-corpus.json](../artifacts/int8_optimization_v4/baseline-canister-corpus.json)。基準Wasmと採用Wasmは同じローカルcanister・同じpackで順番に実行した。warmupメモリは16 tensorごとの観測値で、瞬間的な未観測ピークを含む厳密な上界ではない。計測した追加量は512 MiBの許容幅から十分離れている。実ネットワークの料金・遅延・可用性はこのlocal測定から確定しない。

## 短入力query診断API追加後

後続の短入力調査でowner専用`infer_tokens_query`を追加した。追加直後のWasm hashは`0x6914da6f6d286b659d292307a6fc250df69ad2c71e60a52466a16f09d9128a80`、サイズは5,893,463 bytes。元の96入力との再比較（ローカル生成物: `artifacts/int8_optimization_v4/query-version-vs-baseline-canister.json`）ではlogits最大絶対差0.0、判定96/96一致、全件の命令数が8.301%以上減った。warmupは33,913,236,889命令、観測メモリ最大957,033,623 bytes（記録（ローカル生成物: `artifacts/int8_optimization_v4/short-query-warmup.json`））。旧Choice schemaの単一updateは28 tokensで8,493,557,400命令、128 tokensで39,258,249,055命令となり両方成功した（記録（ローカル生成物: `artifacts/int8_optimization_v4/query-version-update-limits.json`））。

最終的にqueryへ**16-token上限と現行pack hash照合**を追加した。17 tokens以上は`TooLong`、別packは`BindingMismatch`で推論前に拒否する。Wasm hashは`0xdd7013df97f0b2b039540cb94888aebd7239efe085d53935601f6206236d1b49`、サイズは5,894,436 bytes。16-tokenの全18条件が実queryで成功し、最大4,756,310,000命令（境界検証（ローカル生成物: `artifacts/int8_optimization_v4/query-guard-check.json`））。warmupは33,913,836,123命令、観測メモリ最大957,034,596 bytes（記録（ローカル生成物: `artifacts/int8_optimization_v4/guarded-query-warmup.json`））。元の96入力は再比較（ローカル生成物: `artifacts/int8_optimization_v4/guarded-query-vs-baseline-canister.json`）でlogits最大差0.0、判定96/96一致、命令数も全件8.301%以上削減。従来の単一updateは28 tokensで8,494,991,111命令、128 tokensで39,248,061,951命令となり両方成功した（記録（ローカル生成物: `artifacts/int8_optimization_v4/guarded-query-update-limits.json`））。queryの入力例と判断品質上の制約は[短入力メモ](INT8_SHORT_QUERY.md)を参照。
