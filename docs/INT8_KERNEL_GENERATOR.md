# INT8行列積の命令数削減: カーネル生成と実モデル測定

2026-09-25～28。W8A8の整数積を行う`dot_tile`の16バイトKループを生成器に移し、展開数を1・2・3・4・8・16で比較した。命令数では**８回展開**がCIビルド可能な最良候補だった。16回展開はLinux CIのWasmビルドが完了しなかった。８回展開もMacの実モデル推論とLinux x86の合成ベンチで遅くなり、**通常推論への採用を撤回した**。現行コードは元の整数積を使用する。ICの計上方式、モデルpack、量子化、積和順序、出力APIは変更していない。

## 実モデルでの結果

変更前と各候補を**同じソースの他の部分、同じRustツールチェーン、同じINT8 pack**からビルドした。別ポートの独立したlocal ICネットワークに403MiBのpackを一度アップロードし、各Wasmへのupgrade後に201 tensorをwarmupした。96入力は自然文24件×3 schemaと境界8件×3 schema。100 tokens超は既存の分割APIを使い、各入力で総命令数とlogitsを比較した。

| Wasm | 96入力の命令数削減率・中央値 | 最小～最大 | logits最大絶対差 | 判定不一致 |
|---|---:|---:|---:|---:|
| **8回展開（候補）** | **5.613%** | **5.184～5.704%** | **0.0** | **0/96** |
| 16回展開（CIビルド不可） | 6.304% | 6.136～6.658% | 0.0 | 0/96 |

128-token Choice境界入力では、変更前**39,283,922,912**命令から採用した８回展開で**37,179,745,325**命令（5.357%減）。16回展開では36,820,406,517命令（6.271%減）だった。これでもqueryの5B命令上限には届かない。実モデルの生値は[変更前](../artifacts/int8_kernel_generator/real-original.json)、[8回展開](../artifacts/int8_kernel_generator/real-unroll8.json)、[16回展開](../artifacts/int8_kernel_generator/real-unroll16.json)。module SHA-256は順に`0298b36221d5676f7721640b56f15ad7bd7cb81b765fba1f9c88333494007030`、`a0fb59f29325aede13626148f3725196c1c58b53dcc2b164aac5e941c82bab35`、`b83b5d023d0027cb39704fbba3625a94ab7f2b52de12b30046b64810a4342d7e`。pack SHA-256はすべて`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。

## 単体行列積と見送った案

PocketIC v15.0.0のowner専用`benchmark_int8_kernel`を使い、各候補を別canisterにinstallした。同じ決定的なW8A8入力で各形状を準備1回・測定5回実行し、checksum一致を確認した。以下は変更前に対する16回展開の中央値。

| tokens × 出力行 × K | 変更前 | 16回展開 | 削減率 |
|---|---:|---:|---:|
| 28 × 3072 × 1024 | 64,893,163 | 60,477,547 | 6.804% |
| 128 × 3072 × 1024 | 274,433,287 | 252,597,127 | 7.957% |
| 128 × 5248 × 1024 | 464,863,354 | 427,559,914 | 8.025% |
| 128 × 1024 × 2624 | 230,635,527 | 212,917,383 | 7.682% |

[全サンプル](../artifacts/int8_kernel_generator/pocketic-original-unroll16.json)には各呼び出しの命令数、checksum、壁時計時間、Wasm hashを保存した。８回展開は同じ３つの128-token形状で6.54～6.86%減だった（[全サンプル](../artifacts/int8_kernel_generator/pocketic-unroll2-4-8.json)）。入力i8をタイルごとにi16へ事前拡張して再利用する試行は、128-tokenの３形状で**0.20～1.59%増**となり撤回した（[全サンプル](../artifacts/int8_kernel_generator/pocketic-preexpanded-input.json)）。

16回展開のWasmは7,352,703 bytes、gzip時1,867,382 bytes。Mac arm64でのreleaseビルドには3分18秒かかり、`rustc`の観測RSSは約8.2GBだった。８回展開のWasmは6,516,334 bytes。CIのLinux runnerでは16回展開のWasmビルドが約88秒後に終了コード143で停止した（GitHub Actions run 36191742651）。Rustコンパイルエラーは記録されていないが、同じジョブで他のWasmはビルドできており、16回展開のコンパイル資源消費が原因と考えられる。

この計測のPocketIC更新呼び出しの壁時計時間は、16回展開で元より長かった。８回展開も128×3072×1024の合成処理で約0.028→0.042秒に増えた。これはMac上の短い合成処理の値で、実ICノードのCPU負荷を表す測定ではない。

## 実モデルの所要時間（2026-09-28、Mac arm64）

GitHub Actionsの変更前Wasm (`191c1c47…`) と８回展開Wasm (`5dd29878…`) を、同じlocal canisterへ順にinstall・upgradeした。同じ403MiB packを一度uploadし、各Wasmで201 tensorをwarmupした。各入力は準備1回と測定3回を行い、壁時計時間の中央値を比較した。計測には`icp` CLI起動とlocal replica通信の時間も含む。128 tokensは２回のupdateへ分割し、それ以外は単一updateを使った。

| 入力 | 変更前 | ８回展開 | 所要時間の増加 | 命令数の削減 |
|---|---:|---:|---:|---:|
| Choice自然文・44 tokens | 1.811秒 | 1.819秒 | 0.5% | 5.678% |
| Choice境界・64 tokens | 2.323秒 | 3.039秒 | 30.9% | 5.688% |
| Choice境界・96 tokens | 3.438秒 | 4.389秒 | 27.7% | 5.424% |
| Choice境界・128 tokens | 4.819秒 | 6.420秒 | 33.2% | 5.365% |

４入力ともlogitsは完全一致した。生の反復値・Wasm SHA-256・pack SHA-256・corpus SHA-256は[変更前](../artifacts/int8_kernel_generator/full-wall-baseline-linux-ci.json)と[８回展開](../artifacts/int8_kernel_generator/full-wall-unroll8-linux-ci.json)に保存した。再実行には`tools/benchmark_full_model_wall.py --variant NAME --network-root ROOT --expected-wasm WASM --output RESULT.json`を使用する。入力を増やす場合は`--case-id`を繰り返す。

この環境では64 tokens以上の実推論が大幅に遅くなり、当初の「検証入力で1%以上の性能悪化がない」採用条件を満たさない。

## Linux x86の合成ベンチ（2026-09-28）

GitHub Actionsの`ubuntu-latest`でPocketIC v15.0.0を使い、同じ２つのCI生成Wasmを別canisterにinstallした。各形状につき準備１回、測定100回を交互に実行した。checksumは全回一致した。

| tokens × 出力行 × K | 変更前のupdate中央値 | ８回展開 | 所要時間の増加 | 命令数の削減 |
|---|---:|---:|---:|---:|
| 28 × 3072 × 1024 | 24.56ms | 27.09ms | 10.31% | 6.343% |
| 128 × 3072 × 1024 | 48.80ms | 51.01ms | 4.53% | 6.839% |
| 128 × 5248 × 1024 | 75.72ms | 79.69ms | 5.25% | 6.893% |
| 128 × 1024 × 2624 | 42.59ms | 45.38ms | 6.53% | 6.579% |

[集計JSON](../artifacts/int8_kernel_generator/linux-x86-ci-wasm-summary.json)にWasm hash、中央値、範囲、PocketIC serverのCPU時間を保存した。全800サンプルはGitHub Actions [run 36363353651](https://github.com/humandebri/IC-Laya-Standalone/actions/runs/36363353651)の`int8-linux-benchmark` artifactにある。CPU時間は個別呼び出しに対して約10ms粒度で量子化され、形状間で増減が混在するため、この結果からCPU負荷の改善・悪化は断定しない。同時実行時のノード負荷も未測定である。

Macでの実モデル遅延とLinux x86での合成update遅延がともに悪化したため、当初の採用条件に従って８回展開を撤回した。元のkernelを通常推論に戻し、生成器、候補Wasmの測定結果、再測定ツールだけを保存する。

## 再生成と再測定

`tools/generate_int8_dot.py`が`crates/laya-candle/src/int8_dot_generated.rs`を決定的に生成する。生成ファイルは比較候補の記録で、通常ビルドには含めない。比較候補を再ビルドする場合は、候補が組み込まれていたcommit `b30414f`を使う。現行の`int8.rs`は元のSIMD kernelを使用する。

```sh
python3 tools/generate_int8_dot.py --check
python3 tools/generate_int8_dot.py --unroll 16  # 比較候補を生成するとき
```

PocketIC比較は`python3 tools/benchmark_int8_candidates.py --variant baseline=BASELINE.wasm --variant candidate=CANDIDATE.wasm --output result.json`。必要なPython packagesは`pocket-ic==3.1.2`、`ic-py==1.0.1`、任意で`psutil`。`POCKET_IC_BIN`にはPocketIC実行ファイルを指定する。実モデル比較には独立localネットワークに同じpackをupload/warmupし、`tools/compare_int8_model_candidates.py --network-root ROOT --expected-wasm WASM --variant NAME --output result.json [--baseline baseline.json]`を使う。計測の詳細は各JSONのmodule hash・pack hash・corpus hashで照合する。
