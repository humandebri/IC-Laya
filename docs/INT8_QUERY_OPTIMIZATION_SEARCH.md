# INT8 query推論の命令数削減探索

最終版では、128-tokenのINT8 query総命令数が **47,209,698,416→42,129,893,539（10.76%減）**。96入力の最適化前後のINT8 logitsは完全一致し、同じ最終Wasm上のF32も実canister参照と完全一致した。通信量74.82%減を維持し、同じWasmのF32比の命令数増加は約19.38%から **6.94%**へ縮まった。既存の8件の判定差は残る。

## 性能の意味

通信量、命令数、判断の再現性、正解率、実際の待ち時間は分けて評価する。探索前のINT8経路は128 tokensでF32より総handler命令数が19.38%多く、通信は74.82%少なかった。F32との最終判定は88/96件一致。正解ラベルがないため正解率の低下とは断定できないが、元のモデルの判断を再現する能力には差がある。ローカル反復queryの時間はcacheを制御しておらず、計算速度の根拠にはしない。

前回の融合処理自体はINT8 logitsを変えず命令数を0.345%減らした。8件の判定差については[誤差分析](INT8_FUSION_AND_ERROR_ANALYSIS.md)を参照。

## 今回の探索

既存の `feat/client-held-query-inference` ブランチで、固定scale・近似exp・近似GELUを導入せずに命令数の削減を試した。候補ごとに実packの43・65・128-token入力をowner用の通常queryで実行し、元のINT8 logitsとの完全一致を確認する。探索の基準は[前回融合版](../artifacts/int8_fusion/summary.json)、同じpackと入力で比較する。

### attentionの内積をまとめる

QKで各keyとの内積を1本ずつ呼び、AVでも各channelとの内積を1本ずつ呼んでいた。共有入力を持つ最大16本の内積をまとめ、入力のロード・符号拡張を共有する。積和はINT32で、加算の結果は正確な整数のまま。距離maskとSoftmaxのF32計算・集計順は維持する。固定packはlocal attention幅128で入力も128 tokens以下。他packでmaskを狭くする場合は、まとめ計算で不要な内積が増えないか別の評価が必要。AVの結果は同じscale積を使って直接F32の作業領域へ書き、全出力のINT32配列とhead scale配列、別のF32行コピーを整理した。

### SIMDのデータ準備と書き出し

Q/K/VのINT8→scale適用、RoPEの対になった回転、AVのINT32→scale適用をSIMD化する。RoPEは元のF32の乗算・加減算順を保持し、FMAへ置き換えない。位置ごとのsin/cosも元のF64式で作る。INT8出力の4値は飽和narrowでまとめ、4-byte storeで書き出す。量子化の最大値走査、除算、half-away-from-zero、±127 clampは維持する。この書き出し改善は既存F32 activation経路にも共通で使われる。

### 入力だけを先に符号拡張する案

Linearの入力だけを一度i16へ符号拡張し、出力tile間で再利用する。重みはi8のまま、追加作業領域は実packで最大128×2624×2 = 671,744 bytes。過去の重み全体のi16化と違い大きな恒久メモリ増加はないが、今回の3入力では直前の候補から総命令数が約0.4〜0.6%増えたため採用しなかった。準備や追加ロードを含めて測る必要がある。試行ソース（ローカル生成物: `artifacts/int8_search/rejected-preexpanded-int8.rs`）を保存した。

### LayerNormと作業配列

INT8の和・二乗和を1回の走査でSIMD集計する。K≤16384なら各INT32 square laneは最大2^26、sum laneは絶対値最大2^19でoverflowしない。平均・分散・逆数係数は従来のF64計算を維持し、各値のF64正規化→F32 cast→weight/biasをSIMDで処理する。

attentionではQ/K/V、scale、転置V、確率、量子化確率の作業配列をhead間で再利用する。各headで全要素を上書きし、query内のローカルな作業領域として保持する。

### Linearのブロック幅

過去のF32 activation版では16×8、16×16、32×8、32×16、64×8、2回展開などを実測し、64×16の2回展開が採用されている。[既存の試行記録](INT8_V4_REJECTED_TRIALS.md)を先に確認し、今回はgroup scratchとSIMD書き出しを変更したINT8経路で32行ブロックを再比較した。65 tokensで約1.29%、128 tokensで約1.27%増加し、64行を維持した。43-tokenの差はごく小さく、優劣の根拠にしていない。

## 候補の実測

以下はすべて前回の融合版を基準にした総handler命令数。A→B→Dは変更を累積する。CはBに入力の先行符号拡張を追加した試行で、悪化したためDには含めない。EはDの64行tileを32行へ変更した試行。最終列の比較は同じ入力の以前のINT8 logitsに対するもの。

| 候補 | 43 tokens | 65 tokens | 128 tokens | logits |
|---|---:|---:|---:|---|
| A: 内積のまとめ計算・AV書き戻し整理 | 16,021,731,638 | 22,689,321,051 | 45,631,302,580 | 3件とも完全一致 |
| B: A＋SIMD準備・まとめ書き出し | 15,098,522,801 | 21,291,502,331 | 42,856,108,109 | 3件とも完全一致 |
| C: B＋入力の先行符号拡張（見送り） | 15,184,854,342 | 21,387,208,022 | 43,045,414,961 | 3件とも完全一致 |
| D: B＋norm SIMD・配列再利用 | 14,855,935,717 | 20,925,865,494 | 42,127,203,147 | 3件とも完全一致 |
| E: D＋32行ブロック（見送り） | 14,858,032,578 | 21,195,432,240 | 42,661,626,852 | 3件とも完全一致 |

Dの削減率は43 tokensで8.27%、65 tokensで9.87%、128 tokensで10.77%。候補の測定データ（ローカル生成物: `artifacts/int8_search/norm-simd-reuse.json`）を保存した。

## 検証の範囲

実際のWasmをNodeで実行する検証では、整数積和920条件、量子化144条件、F32書き戻し216条件、INT8準備・RoPE・norm144条件、統計集計24条件を検査する。準備・normはscalar参照とF32のbit単位で比較し、量子化は丸めの境界値と長さの端数も含む。統計は−128・127や最大Kの正確なi64参照と比較する。

最終候補は96入力についてF32・INT8を同じWasm上で対にして実行し、既存の実canister参照と以前のINT8 logitsとの完全一致を確認した。判断の一致は正解ラベルに対する精度評価の代用にはしない。

同一Wasmを再upgrade/warmupしたDの再測定では、128-tokenの命令数に約0.006%の変動があった。小さな差は実行状態による変動も含み得るため、43-tokenでのtile差のような微小な数値を改善・悪化の根拠にはしない。壁時計の時間差を命令数削減へ換算しない。

## 最終確認

| 指標（128 tokens・3 markers・2段/query） | F32 activation | INT8 activation |
|---|---:|---:|
| 総handler命令数、交互に5回の中央値 | 39,396,527,412 | 42,129,893,539 |
| 通信集計、Candidリクエスト＋返信continuation | 16,796,736 bytes | 4,230,208 bytes |
| 推論query回数 | 17 | 17 |

96件すべてで命令数が減り、削減率は7.92〜10.76%、入力ごとの中央値は8.62%だった。これは同じ入力を前回の融合版と比べた値で、入力長だけを変えた比較ではない。最大query handler命令数は3,138,344,891。F32とINT8の判定一致は88/96件で、最適化前と同じ8件だけが異なる。

最終Wasmは `0xe21d3ae54de974072fa401be39cbe99d2e07b8daf01e37477764aa4135653657`、packは `bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。Rust32 tests、Python75 tests、既定featureのworkspaceテスト、Wasm参照検査1,448条件を通過した。実canisterで不正状態・scale・step数の拒否、owner制限、同じqueryの再試行、update jobの非変更を確認した。CanisterのAPI・continuation形式・quantization revisionは維持する。

[集計](../artifacts/int8_search/summary.json)、全96入力（ローカル生成物: `artifacts/int8_search/final-corpus.json`）、交互5回の比較（ローカル生成物: `artifacts/int8_search/final-benchmark-128.json`）、検証記録（ローカル生成物: `artifacts/int8_search/checks.json`）を保存した。候補の各変更の効果を個別に完全分離した測定ではなく、表は累積した構成の比較。短い入力の微小なtile差を除き、命令数の減少が大きく、logitsを維持できたD構成を採用した。

演算別profile（ローカル生成物: `artifacts/int8_search/norm-simd-reuse-profile.json`）では、attention中核は約87.7億→48.2億（約45%減）、normは約11.83億→4.45億（約62%減）、Linearの出力量子化は約14.00億→11.37億命令へ減った。spanはinclusiveで`q8.linear`に出力量子化・積和が含まれるため、二重計上しない。profileは同じ最終Wasmの初回warmup時の記録で、各counterの上乗せがあり、通常queryの総命令数とは別に扱う。

主要な残余コストはLinearの積和・書き戻し（約301.7億命令）。入力の先行符号拡張や32行tileは今回の実装では改善しなかった。固定scaleやQ31、近似exp/GELUも自動採用していない。

再現する場合、既にモデルをuploadした専用local canisterをCandle有効のWasmへupgradeしてwarmupする。`tools/measure_quantized_candidate.py`で3入力の候補比較、`tools/validate_quantized_queries.py`でF32との対の96件検証、`tools/measure_activation_queries.py`で5回の比較を実行する。新しい測定は別の出力パスに保存し、歴史的な結果を上書きしない。warmupとmodule hash確認は推論queryとは別の準備・検証操作で、推論自体のupdate切り替えはない。

検証に必要な96入力の参照logitsは[コンパクトな参照データ](../artifacts/references/int8-final-corpus.json)として保存する。queryごとの詳細記録は含めず、元記録のSHA-256で測定履歴に結び付ける。
