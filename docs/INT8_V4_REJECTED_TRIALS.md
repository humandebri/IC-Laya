# Laya INT8 V4: 採用しなかった試行

2026-09-25のlocal canister試験の記録。比較の基準と採用結果は[INT8_OPTIMIZATION_V4.md](INT8_OPTIMIZATION_V4.md)。以下の「増加」は命令数の悪化を指す。候補ソースと測定JSONは`artifacts/int8_optimization_v4/`に残してある。checkpointとpackは全試行で同一。

## weight配置

| 試行 | 測定結果 | 見送り理由 | 記録 |
|---|---|---|---|
| warmupで符号拡張したi16を出力ブロック配置 | 3形状×5長の行列積で中央値**+5.120%**。実モデル128-token Choiceは**44,781,317,859命令**（基準42,841,760,152命令）。warmupは33,913,718,557→39,289,848,362命令。観測メモリ最大値は956,831,731→1,424,420,991 bytes。 | 推論とwarmupの両方が遅く、追加前処理を回収する推論回数が存在しない。メモリ増分は約446 MiBで許容512 MiB以内だが、性能条件を満たさない。 | 行列積（ローカル生成物: `artifacts/int8_optimization_v4/i16-matrix.json`）、実モデル（ローカル生成物: `artifacts/int8_optimization_v4/i16-choice-128.json`）、warmup（ローカル生成物: `artifacts/int8_optimization_v4/i16-warmup.json`）、ソース（ローカル生成物: `artifacts/int8_optimization_v4/i16-int8.rs`） |
| 出力ブロック単位のcompact i8配置 | 行列積15条件の中央値**+4.309%**、全条件で+2.664〜+4.801%。checksumは基準と一致。 | 読み出し配置を変えても、このWasm kernelでは命令数が増えた。 | 行列積（ローカル生成物: `artifacts/int8_optimization_v4/blocked-i8-matrix.json`）、ソース（ローカル生成物: `artifacts/int8_optimization_v4/blocked-i8-int8.rs`） |

i16試行では線形層で不要になった元weightのヒープコピーを解放する構成も試した。採用版のweight配置・pack形式は基準のまま。

## 行列積タイル

各候補を3072×1024、5248×1024、1024×2624の3形状と28・38・64・96・128 tokensの計15条件で測定した。表は基準64×16に対する中央値。準備1回、測定3回の生値・範囲・checksumはリンク先JSONにある。全候補でchecksumは基準と一致した。

| タイル | 中央値の命令数変化 | 判断 | 記録 |
|---|---:|---|---|
| 16×8 | +8.770% | 全体として遅い | JSON（ローカル生成物: `artifacts/int8_optimization_v4/tile16x8-matrix.json`） |
| 16×16 | +3.488% | 一部条件で最大0.507%改善しても、多くで悪化 | JSON（ローカル生成物: `artifacts/int8_optimization_v4/tile16x16-matrix.json`） |
| 32×8 | +6.140% | 全体として遅い | JSON（ローカル生成物: `artifacts/int8_optimization_v4/tile32x8-matrix.json`） |
| 32×16 | +0.983% | 一部条件で最大0.530%改善しても、多くで悪化 | JSON（ローカル生成物: `artifacts/int8_optimization_v4/tile32x16-matrix.json`） |
| 64×8 | +4.813% | 全条件で悪化 | JSON（ローカル生成物: `artifacts/int8_optimization_v4/tile64x8-matrix.json`） |
| 32×16、内側ループ2回展開 | -7.799% | 改善するが、64×16の2回展開（-8.893%）は64 tokens以上の9条件で0.981〜1.769%速い。28・38 tokensの6条件では32×16が19命令だけ少ない | JSON（ローカル生成物: `artifacts/int8_optimization_v4/tile32x16-unroll2-matrix.json`）、ソース（ローカル生成物: `artifacts/int8_optimization_v4/tile32x16-unroll2-int8.rs`） |

形状別のタイル選択は加えていない。64 tokens以上では主要3形状すべてで64×16が勝ち、28・38 tokensでは64行タイルに到達せず既存の小タイル経路を使う。128行タイルは今回の計画どおり再試行していない。

## attention

| 試行 | 測定結果 | 見送り理由 | 記録 |
|---|---|---|---|
| scale・mask・softmaxを行単位で連続処理 | 代表128-token入力では2回展開のみの39,850,355,553→39,724,256,181命令（約0.316%改善）、logits一致。ただしnativeの96入力互換比較では最大絶対差**0.6466925**、判定**2件不一致**。 | 許容差0.002と全件判定一致の両方を満たさない。代表入力だけでは検出できなかった。近似expは使っていない。 | 代表入力（ローカル生成物: `artifacts/int8_optimization_v4/softmax-choice-128.json`）、96入力比較（ローカル生成物: `artifacts/int8_optimization_v4/softmax-corpus-comparison.json`）、ソース（ローカル生成物: `artifacts/int8_optimization_v4/softmax-lib.rs`） |
| QK専用Wasm SIMD | 128-token Choiceで40,780,221,455命令。QKの通常経路と比較した39,135,954,329命令より約4.20%遅い。代表入力のlogits最大絶対差は約0.110。 | 性能と誤差の両条件を外れた。 | 実モデル（ローカル生成物: `artifacts/int8_optimization_v4/qk-simd-choice-128.json`）、ソース（ローカル生成物: `artifacts/int8_optimization_v4/qk-simd-lib.rs`） |
| AV専用Wasm SIMD | 128-token Choiceで45,538,387,672命令。採用候補39,135,954,329命令より約16.36%遅い。代表入力のlogitsは一致。 | 誤差は許容できても性能が大幅に悪化した。 | 実モデル（ローカル生成物: `artifacts/int8_optimization_v4/av-simd-choice-128.json`）、ソース（ローカル生成物: `artifacts/int8_optimization_v4/rejected-attention-simd.rs`） |

QKの転置Kから不要な`contiguous()`を除く変更は採用した。上のattention試行を戻した最終版と、全96入力の実canister比較を[採用結果](INT8_OPTIMIZATION_V4.md)に記録している。
