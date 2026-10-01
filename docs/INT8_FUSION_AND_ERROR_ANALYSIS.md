# INT8融合処理と判定変更の分析

## 比較基準の訂正

前回の「7/96件の判定変更」は、`baseline-corpus-logits.json`という別の参照結果との比較だった。同じpackを使った実canisterのF32結果 `baseline-canister-corpus.json` を基準にすると **8/96件が変更、88/96件が一致**する。従来の7件はすべて含まれ、`score-boundary-112` が追加される。過去の生データは保存し、比較基準を訂正した記録（ローカル生成物: `artifacts/int8_fusion/rebased-initial-corpus.json`）で参照の違いを明記した。

F32比較基準はpackと96入力のハッシュに結び付いている。変更ケースの診断では現在のF32 queryも実行し、基準との差が全8件で0だった。新しい検証ツールは、同じWasm上でF32とINT8を対にして実行する。

## 変更ケース

表のmarginは、F32の上位2 logitsの差。Encoder差は28層後の中間状態の相対L2誤差。正解ラベルのない互換性テストなので、どちらの判定が正しいかはこの比較から判断できない。

| 入力 | F32 → INT8 | margin | Encoder差 |
|---|---|---:|---:|
| choice-natural-03 | reject → review | 0.063826 | 35.6% |
| choice-natural-11 | review → reject | 0.404196 | 19.8% |
| choice-natural-23 | reject → review | 0.265124 | 45.1% |
| choice-boundary-35 | review → refund | 0.484633 | 33.7% |
| choice-boundary-96 | refund → review | 0.137803 | 47.1% |
| score-natural-07 | high → medium | 0.032949 | 28.6% |
| score-boundary-96 | high → medium | 0.035644 | 28.9% |
| score-boundary-112 | medium → high | 0.004472 | 33.4% |

元の7件のうちmarginが0.07未満なのは3件。追加された1件を含めると4/8件である。残りは0.138〜0.485あり、単なる僅差では説明できない。`boundary` はテキストを繰り返して指定長へ切り詰めた入力で、独立した自然文ではない。

## 誤差が生じる場所

同じ入力について、各段の中間状態を採取し、F32状態を量子化してINT8後段へ、INT8状態を復元してF32後段へ渡した。**全8件で、EncoderをF32にした場合はINT8後段でも元の判定に戻り、EncoderをINT8にした場合はF32後段でも戻らなかった。** final norm・qtype加算後、scorer直前で切り替えても同じ傾向だった。Encoder側の累積誤差が主要因と考えられる。単独の演算を原因として特定したわけではない。

Embedding直後の相対L2差は約2%だが、28層後には約20〜47%へ増える。誤差は層ごとに単調増加しない。各段の入力をF32から量子化し直して単独で評価した診断でも、encoder layer 7（完了段8）などに大きい差があり、後段scorerだけをF32にしても解消しなかった。残差、norm、QKV/RoPE、Softmax/AV、MLPの再量子化を個別に外す診断は未実施。

単独の段を評価したときの大きい相対L2差は次の通り。完了段1はEncoder layer 0に対応する。この値はF32入力をその都度量子化し直した診断で、通常推論中の累積差とは区別する。

| 入力 | 大きい3段（完了段: 相対L2差） |
|---|---|
| choice-natural-03 | 8: 8.92%, 6: 8.02%, 3: 7.95% |
| choice-natural-11 | 8: 9.57%, 3: 7.92%, 4: 7.82% |
| choice-natural-23 | 8: 8.00%, 3: 7.83%, 4: 7.77% |
| choice-boundary-35 | 8: 9.97%, 6: 8.60%, 3: 7.95% |
| choice-boundary-96 | 8: 10.25%, 6: 8.29%, 3: 8.04% |
| score-natural-07 | 13: 9.66%, 8: 8.74%, 6: 8.47% |
| score-boundary-96 | 6: 8.64%, 8: 8.34%, 13: 8.08% |
| score-boundary-112 | 6: 8.71%, 8: 8.62%, 3: 8.15% |

各段の誤差と切り替え診断（ローカル生成物: `artifacts/int8_fusion/change-analysis.json`）に全8件を保存した。これは融合前のWasm `7f82edaf…` の診断である。

## 処理をまとめる変更

Linearで `R×全出力` のINT32配列を作り、その後F32の行へ書き戻していた処理を変更した。最大64×16のINT32計算tileから直接scaleとbiasを適用し、再利用する出力groupのF32 scratchへ書く。groupが揃ったら動的scaleを決め、INT8へ量子化する。計算順序・量子化単位・丸め方は維持する。

実packのQKVは48 groups（16 heads×Q/K/V）、group幅64。64-token tileのINT32一時配列786,432 bytesと行scratch12,288 bytesの代わりに、16,384 bytesのgroup scratchと最大4,096 bytesのINT32計算tileを使う。MLPのwiは2 groupsなのでscratchは671,744 bytes残る。全演算のF32 scratchをなくしたという意味ではない。

動的scaleを保つ限り、group内の最大値が判明する前に最終INT8は確定できない。したがって最大値走査とgroup scratchは残す。行列の積和数は同じで、削減対象は配列確保・書き戻し・再読み込みなどの周辺命令である。

### 融合後の全体測定

128 tokens・3 markers・2段/query、同じ新Wasm上でF32/INT8を交互に5回実行した。総handler命令数はF32 **39,544,344,114**、INT8 **47,209,698,416**。融合前のINT8 47,373,198,979から **0.345%減**だが、F32比ではまだ **19.38%増**する。通信は16,796,736→4,230,208 bytesで **74.82%減**を維持した。

演算別profileでは融合した積和・書き戻しが約301.7億、attention中核が約87.7億、出力量子化が約14.0億命令だった。総命令の大部分は積和とattentionに残り、配列の削減だけで計算全体を大きく軽くすることはできない。

96件すべてで融合前後のINT8 logitsは完全一致し、新WasmのF32 queryもpack/corpusに結び付いた実canister参照と完全一致した。判定差は訂正後の8件のまま。F32とINT8の最大logit差は3.1853323、入力ごとの最大差の中央値は0.2793638。最大query handler命令数は3,446,430,957だった。

新Wasmは `0x1abfbec7c96856f86bd8bbf86f891d4a480716c3a96fa6c9b1f3491630becc50`、packは `bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。[集計](../artifacts/int8_fusion/summary.json)、96入力（ローカル生成物: `artifacts/int8_fusion/corpus.json`）、反復測定（ローカル生成物: `artifacts/int8_fusion/benchmark-128.json`）、演算別profile（ローカル生成物: `artifacts/int8_fusion/profile-128.json`）に保存した。Rust31 tests、Python75 tests、既定featureのworkspaceテスト、Wasm buildが通過した。protocol検証を含む確認記録（ローカル生成物: `artifacts/int8_fusion/checks.json`）も保存した。

## 固定scaleと整数シフトの実験

固定scaleなら最大値走査を省き、INT32→scale/bias→丸め→INT8を書き出せる。ただし入力scale `sx` はtokenごと、重みscale `sw` は出力channelごとなので、出力scale `sy` を固定しても係数 `sx×sw/sy` 全体が演算ごとに一つの定数になるわけではない。

診断用query `benchmark_q8_epilogue(blob)` と再現ツール `tools/benchmark_quantized_epilogue.py` を追加した。実モデルの最初のEncoder QKVのINT32積和を使い、動的scaleのstaged SIMD、固定scaleの融合F32 SIMD、固定scaleのQ31乗算・シフト SIMDを比較する。行列積自体とモデル全体の精度はこのbenchmarkの対象外。整数係数の準備コストを別記し、tokenごとの係数計算は処理命令数に含める。

Q31ではINT32合計×INT32係数をINT64で保持し、biasと丸め定数を加えて31 bitシフトし、±127へclampする。範囲検証後に演算する。F32式と係数近似・丸め順序が異なるため完全一致は保証しない。固定scaleとQ31は通常のモデル推論には組み込まず、明示的なowner用実験APIに限定する。

最初のQKVだけについて、3 schema×8自然文の24入力（token列は24種類）でhead別scaleを調整し、残り72件をholdoutにした。モデル全層を較正した結果ではなく、固定scaleを自動採用する根拠にはしない。

### epilogue実測

| 入力 | 動的scale staged | 固定scale F32融合 | 固定scale Q31融合 | Q31係数の準備 |
|---|---:|---:|---:|---:|
| 43 tokens | 6,196,608 | 4,879,822 | 24,088,879 | 690,725 |
| 128 tokens | 18,399,504 | 14,267,507 | 71,745,866 | 673,322 |

単位はWasm命令数。処理配列の確保を含む。入力のCandidデコード、INT32行列積、検証用誤差集計は計測区間外。準備は毎回必要ならQ31列に加算する。浮動小数点融合ではsyによる除算と丸めを直接SIMDで行う。

固定scaleのF32融合は、このepilogueだけで約21〜22%減った。一方、Q31融合は準備を含める前から動的方式の約3.9倍だった。INT64積、丸め、clamp、tokenごとの係数作成などを含むため、この実装・ICの命令計測では整数化が割に合わなかった。整数乗算・シフト一般が常に遅いという結論ではない。

24件で最大値を較正した固定scaleでは、holdout72件の12,681,216値中135値がclampされた。固定scaleでclampされた判定変更ケースは `choice-natural-03`、`choice-natural-11`、`choice-natural-23`、`score-natural-07`。これは最初のQKV出力の観察で、判定変更の原因をこのclampへ帰属させるものではない。元の動的scale実験で既に変更が起きている。

43-token入力では動的方式のRMSE 0.008187に対し固定方式0.013313、最大誤差0.032821→0.256699。128-token入力でもclampはなかったがRMSEは0.008561→0.013270に増えた。大きめの固定scaleはclampを減らす代わりに量子化の刻みを粗くする。整数Q31は固定F32方式に対し43-tokenの132,096出力中1値が違い、128-tokenでは0値だった。F32との一致を単純に保証できる方式ではない。

較正・holdout・実測結果（ローカル生成物: `artifacts/int8_fusion/epilogue.json`）を保存した。全層の固定scaleを採用するには代表入力による全層較正、clampと累積誤差の検査、正解ラベルでの評価が必要になる。今回の固定scaleとQ31は診断用に留め、動的scaleを保つ配列削減だけをINT8実験経路へ採用した。

## 測定の再現

専用local canisterをCandle有効でbuild・upgradeし、既存packをwarmupした後に実行する。モデルのuploadは不要。

```bash
python3 tools/validate_quantized_queries.py --project-root build/client-query-project --output artifacts/int8_fusion/corpus.json
python3 tools/profile_quantized_queries.py --project-root build/client-query-project --input artifacts/laya-choice-128-input.json --output artifacts/int8_fusion/profile-128.json
python3 tools/measure_activation_queries.py --project-root build/client-query-project --input artifacts/laya-choice-128-input.json --output artifacts/int8_fusion/benchmark-128.json
.venv/bin/python tools/benchmark_quantized_epilogue.py --output artifacts/int8_fusion/epilogue.json
.venv/bin/python tools/analyze_quantized_changes.py --project-root build/client-query-project --corpus-results artifacts/int8_fusion/corpus.json --output build/current-change-analysis.json
```

profileのspanはinclusive。`q8.linear` は `q8.fused_dot_writeback` と `q8.output_quantize` を含み、足し合わせると二重計上する。全queryのinstructionsはhandler内のみでCDK引数デコード・返信エンコードは含めない。ローカル時間はcacheを制御せず、計算速度の改善とは解釈しない。epilogue診断queryはowner限定で入力サイズ・shape・有限値・整数演算の範囲を検証し、canister内のモデルやjobを変更しない。

その後の[命令数削減探索](INT8_QUERY_OPTIMIZATION_SEARCH.md)では、同じINT8 logitsを維持し、融合版からさらに約10.8%減らした。
