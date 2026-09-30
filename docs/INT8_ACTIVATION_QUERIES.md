# INT8の中間状態を保持するquery推論

2026-09-30、`feat/client-held-query-inference`ブランチに実験用のINT8 activation経路を追加した。重みだけでなく、演算間・query間の中間状態もINT8とscaleで保持する。従来のF32 activation経路は既定のまま残している。

> 比較基準の訂正: 以下の旧精度集計は別の参照logitsを用いていた。同じpackの実canister F32結果に対しては **88/96件一致・8件変更**。融合後の測定と変更原因は[追加分析](INT8_FUSION_AND_ERROR_ANALYSIS.md)を参照。以下の数値と処理説明は融合前の履歴として保存する。

> 最新の命令数最適化は[INT8 queryの削減探索](INT8_QUERY_OPTIMIZATION_SEARCH.md)を参照。融合版からさらに約10.8%減り、同じWasmのF32比は約6.9%増へ縮まった。

## 実モデルの測定結果

同じ最終Wasm、同じpack、128 tokens・3 markers・2段/queryで、F32とINT8の経路を交互に5回測定した。

| 指標 | F32 activation | INT8 activation |
|---|---:|---:|
| query回数 | 17 | 17 |
| Candidリクエスト＋返信continuation | 16,796,736 bytes | 4,230,208 bytes |
| 総命令数の中央値 | 39,544,882,944 | 47,373,198,979 |
| ローカル反復時間の中央値 | 2.282秒 | 1.015秒 |

通信データの集計値は**74.82%減**、handlerの総命令数は**19.80%増**だった。反復時間にはquery cacheの影響が入り、計算そのものが高速化したことを示さない。最初の比較では6.658秒と5.647秒、その後は両方とも短くなった。cacheを制御した本番ネットワークのlatency測定は行っていない。

旧集計では別の参照結果と最終判定が一致したのは**89/96件**（実canister F32基準では88/96件）。logitsの最大絶対差は3.9964833、入力ごとの最大絶対差の中央値は0.29853605だった。したがって、通信量は削減できたが、元のモデルと同じ判断を維持する経路としては採用できない。既定の推論やReceipt用推論は変更せず、`--query-int8`で明示的に選ぶ実験経路とした。

命令数の内訳では、整数Linearの積和・INT32一時領域が約301.6億、再量子化が約16.7億、attentionの中核処理が約87.7億、normが約11.9億だった。整数化による通信削減と、再量子化・attentionの演算コストは別に評価する必要がある。

再量子化の書き戻しをWasm SIMDで処理し、128-tokenの総命令数を約514.1億から約473.7億へ減らした。SIMD化前後の96入力のlogits最大差は0.0。測定した96入力で1 queryのhandler命令数は最大3,457,108,905だった。この値は測定範囲の結果であり、対応する全入力の上限保証ではない。

最終Wasmは`0x7f82edaf3ca92b86d186f89a842eb3a314a7979de9af7f814f972c955c479c05`、packは`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。[集計](../artifacts/int8_activations/summary.json)、反復比較（ローカル生成物: `artifacts/int8_activations/benchmark-128.json`）、演算別profile（ローカル生成物: `artifacts/int8_activations/profile-128.json`）に記録した。

分割幅と可逆圧縮の候補比較は[queryの通信・分割条件](QUERY_COMMUNICATION_CONDITIONS.md)、実装後のWasm計測は[自動分割と可逆圧縮](QUERY_TRANSPORT_IMPLEMENTATION.md)を参照。現在のCLIは分割幅の自動選択と可逆圧縮を既定で使う。以前の生データによる比較を再現する場合は、固定幅に加えて`--query-compression none`を指定する。

## 実行方法

Candleを有効にしてbuildし、モデルをwarmupした後に実行する。

```bash
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine
python3 tools/canister_infer.py \
  --identity ic-laya-int8 \
  --input artifacts/laya-choice-128-input.json \
  --query-stepped --query-int8 --query-steps-per-call 2 \
  --output build/int8-query-result.json
```

`--query-int8`を省けば、従来のF32 activationによる分割queryになる。実験経路のbegin/continue/profileは通常queryで、updateへの自動切り替えはない。CLIの準備操作とmodule hash確認は従来どおり別の呼び出しを伴う。

## 計算の変更

通常経路ではLinearがF32 TensorからINT8を作り、整数行列積の出力をF32 Tensorへ戻す。実験経路は既に量子化した入力を整数kernelへ直接渡す。出力はINT32のtileに保持し、1行ずつscaleとbiasを適用してINT8へ再量子化する。Linearの出力全体をF32 Tensorとして確保しない。

LayerNormではINT8入力の和と二乗和から平均・分散を求め、係数を1行ずつ適用してINT8へ量子化する。残差加算は各入力のscaleを反映し、加算結果を再量子化する。GELUはCandleと同じerf式をscalarで評価する。

attentionのQKとAVもINT8の整数積和で計算する。QKVの各headを別scaleにし、RoPEを適用したQ/Kはtoken・headごと、Vはheadごとに量子化する。Softmaxの確率はINT8へ量子化してAVへ渡す。MLPのgate/valueも別scaleで保持する。

全演算が整数だけになるわけではない。scale、bias、normの係数、RoPE、Softmax、GELU、再量子化の一時領域には浮動小数点を使い、行列積の積和とattention出力の一時領域にはINT32を使う。省いたのは中間状態全体のF32 Tensorへの復元と、その状態での保存・通信である。1行、またはattentionの1headのF32一時領域は残る。

## APIと状態形式

```text
begin_token_inference_int8_query(TokenInput) -> Result<QueryInferenceProgress>
continue_token_inference_int8_query(blob, max_steps: nat32) -> Result<QueryInferenceProgress>
profile_token_inference_int8_query(blob, max_steps: nat32) -> Result<ProfiledQuery>
```

owner限定、測定対象の固定pack限定。開始はEmbedding、継続は指定した1〜16段を計算する。実際に通る幅は入力と形式ごとに測定し、命令上限に収まる条件を選ぶ。canisterのupdate用jobに触れず、状態はクライアントが保持する。最後のlogitsはF32で返す。実験結果から自動的にReceiptを発行したり、executorを認可したりする経路はない。

状態はmagic `LAYI`、format version 1、inference revision 2を使う。60-byte headerの配置とtoken/marker列は[従来のcontinuation](CLIENT_HELD_QUERY_INFERENCE.md)と同じ。その後に各行のF32 scaleを並べ、最後にrow-majorのINT8値を並べる。行数は通常token数、最後のdecision層後はmarker数。scaleは有限・正・最大値を制限し、入力・bundle・revision・進行位置・厳密なバイト長を検証する。

F32状態`LAYQ`とINT8状態`LAYI`は相互に受け付けない。初期試行のrevision 1とも互換性はない。モデル差し替えや実験版更新の後は開始からやり直す必要がある。

## 検証

整数Linearの参照計算との比較、連続状態と毎段復元する状態の一致、prenorm/postnormのhead、各qtype、不正状態、scale、長さ、異なる形式の拒否、独立sessionの再試行をテストした。これはINT8経路内部の整合性を確認するもので、F32経路との数値一致を意味しない。

実モデルの互換性検証には既存の96入力とF32 activationのlogitsを使う。判定一致率は元の経路との一致を表し、正解ラベルに対する精度ではない。初期試行は96件中6件で判定が違い、初期結果（ローカル生成物: `artifacts/int8_activations/initial-revision1/corpus.json`）を保存した。QKV・MLPの量子化単位を分けた修正版の結果は96入力の記録（ローカル生成物: `artifacts/int8_activations/corpus.json`）に保存する。

Rustの28 tests、Pythonの75 tests、既定featureでのworkspaceテスト、Wasm buildを通過した。実canisterでもINT8プロトコル検証（ローカル生成物: `artifacts/int8_activations/protocol.json`）と従来経路の回帰検証（ローカル生成物: `artifacts/int8_activations/f32-regression/validation.json`）を行った。従来経路は8条件で直接updateと分割queryのlogitsが完全一致した。

再現例（既にwarmupした専用local canister）:

```bash
python3 tools/validate_quantized_queries.py \
  --project-root build/client-query-project \
  --identity ic-laya-query-local-test \
  --output build/int8-corpus.json
python3 tools/profile_quantized_queries.py \
  --project-root build/client-query-project \
  --identity ic-laya-query-local-test \
  --input artifacts/laya-choice-128-input.json \
  --output build/int8-profile.json
python3 tools/measure_activation_queries.py \
  --project-root build/client-query-project \
  --identity ic-laya-query-local-test \
  --input artifacts/laya-choice-128-input.json \
  --output build/int8-benchmark.json
```

profileの`q8.linear`には`q8.integer_dots`と`q8.requantize`が含まれる。合計時に二重計上しない。queryのinstructionsはCDKの引数デコード・返信エンコードを含まない。通信バイト数は成功したCandidリクエストと返信のcontinuationを数え、最終logitsや返信のCandid envelopeは含まない。ローカルの時間にはCLI・通信・パースを含み、反復queryのcacheは制御していない。
