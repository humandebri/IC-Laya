# int8の追加最適化と2 update推論

> この文書は16×16タイル・SIMD量子化までの履歴。最新のF32書戻し最適化と単一update境界は
> [INT8_F32_WRITEBACK.md](INT8_F32_WRITEBACK.md)を参照。

2026-09-23 JST、ローカルcanister `4caro-hl777-77775-aaaba-cai` で計測した。
同じ実LayaのW8A8 pack、128-token Choiceで、**59.420B → 46.632B命令（追加21.5%削減）**となった。
開始処理を最初のバッチに統合し、呼出しも**3 update → 2 update**に削減した。1Bは10億命令。
入力・packのSHA-256と返却logitsは従来版と一致している。

## 検討した4項目

| 項目 | 実測と採用判断 |
|---|---|
| INT8行列積のタイル拡大 | 4×4、8×8、16×8、16×16を比較し、16×16を採用。128 tokensの主要3形状で、量子化を含む線形層全体を約19〜20%削減 |
| token端数処理 | 16/8/4/2/1行のタイルを使い分ける。さらに余り3行だけを4行へ補完。87 tokensで追加約0.2〜0.7%削減。attentionの入力長は変えない |
| activation量子化のSIMD化 | 最大絶対値・有限値確認・除算・丸めを4要素ずつ実行。単体量子化区間は約79%削減。F32の除算とhalf-away-from-zeroを維持 |
| 開始と最初のバッチの統合 | `start_token_inference_batch`を追加。開始＋16ステップ、残り16ステップの2 updateで実モデルが完走 |

タイルの数値は「token行数×出力行数」。積和は従来どおりi32で、K≤16384なら
`16384*128*128 < i32::MAX`となる。F32 scaleを掛ける順序も保持した。
SIMD量子化ではties-to-evenの`nearest`や逆数乗算を使わず、整数部と小数部から丸める。
これにより、0.5の直前の値を`abs(x)+0.5`で誤って切り上げる問題も避ける。

## 候補ごとの命令数

owner専用の`benchmark_int8_kernel`で同じ合成weight・activationを生成し、実際の
`Int8Matrix::forward`を計測した。以下は128 tokens、単位は百万命令。
列はそれぞれ`[output_features,input_features]`。weight生成とchecksum計算は計測区間外。

| 候補 | QKV [3072,1024] | MLP up [5248,1024] | MLP down [1024,2624] |
|---|---:|---:|---:|
| 従来4×4 | 424.393 | 708.197 | 384.341 |
| 8×8 | 373.779 | 621.731 | 339.208 |
| 16×8 | 357.509 | 593.939 | 325.795 |
| 16×16 | 341.818 | 567.133 | 312.884 |
| 16×16＋SIMD量子化 | **323.736** | **549.088** | **266.403** |

比較用カーネルは `.venv/bin/python tools/benchmark_int8_kernels.py --label NAME` で実行する。
各候補で85/86/87/88/128 tokensの15形状を測定し、baselineとの出力checksum一致を確認した。
端数補完だけの比較は87 tokensの3形状で行った。候補ごとのJSON、ソースとSHA-256、
module hashは[実験記録](../artifacts/int8_optimization_v2/)に保存した。
タイル拡大とSIMD量子化を合わせた主要3形状の削減率は22.5〜30.7%。
この合成カーネルの削減率と、全モデルの21.5%は区別する。

## 実モデルの2 update実行

| 呼出し | 処理 | 命令数 |
|---|---|---:|
| 1 | 開始＋ステップ1〜16 | 24.857B |
| 2 | ステップ17〜32 | 21.774B |
| 合計 | 128-token Choice | **46.632B** |

最大バッチは40Bに対して約37.9%の残予算。ローカルwall timeは9.35秒で、CLI通信を含む。
測定回数・負荷を統制した速度比較ではなく、mainnetのレイテンシを表す値でもない。
モデルのアップロードとwarmupは上表に含まない。命令数はAPI内部の計測区間であり、
Candid処理などの全メッセージ費用を網羅するものではない。
実測JSON（ローカル生成物: `artifacts/int8_optimization_v2/final-choice-128.json`）。

`--stepped`は既定で16ステップずつ。`--steps-per-call 1`は開始＋1ステップずつの旧方式、
`--profile`も1ステップずつ計測する。別モデルでは16ステップが40B以内とは限らないため、
必要なら`--steps-per-call`を小さくする。入力上限`MAX_TOKENS=128`は変更していない。

開始APIは32-byte `request_id`を受け取り、現jobと同じID・入力・バッチサイズの再送には
最初の応答を返す。入力またはバッチサイズを変えて同じIDを使うと`IdConflict`となる。
完了後の開始再送も現在の完了状態を巻き戻さない。保存するのは現jobのみで、
別IDで新規開始、model交換、upgrade後に旧IDの再送を保証しない。
CLIはIDを呼出し前に表示し、`--request-id HEX`で指定できる。
後続バッチの再送保証は直前の開始位置・サイズに限るため、多数の小バッチへ分けた場合に
CLI全体を最初から再実行して任意の過去ステップを再送できるわけではない。
再送・競合・権限の実機検証（ローカル生成物: `artifacts/int8_optimization_v2/final-protocol.json`）。

## 全層プロファイルと残るボトルネック

同じ128-token入力を1ステップずつ計測した比較でも、59.747B → 46.922B（21.5%削減）となった。
これは計測用APIの値であり、上の通常2 update実行46.632Bとは分けて扱う。
区間比較JSON（ローカル生成物: `artifacts/int8_optimization_v2/profile-comparison.json`）で入力・pack・logits一致と、
区間合計が全体命令数を超えないことを確認した。

| 区間 | 従来4×4 | 今回 | 今回の全体比 |
|---|---:|---:|---:|
| INT8行列積・scale適用 | 46.397B | 36.723B | 78.3% |
| activation量子化 | 3.875B | 0.795B | 1.7% |
| attention QK＋AV | 3.687B | 3.675B | 7.8% |
| softmax | 1.722B | 1.723B | 3.7% |

量子化の費用は約79.5%減った。追加改善の中心は引き続きINT8行列積で、量子化や
入力コピーだけをさらに削っても全体への効果は小さい。16×16を超えるタイルや別のweight配置は
今回未測定であり、さらなる削減率は主張しない。浮動小数点attentionの演算順変更や近似化も未実施。

## 単一updateの境界

同じ質問・選択肢を保持して本文tokenを反復・短縮した系列で、長さ別に`infer_tokens`を測定した。
結果は次のとおり。**最大成功実測は109 tokens**で、旧4×4版の88から伸びた。

| 系列 | 108 tokens | 109 tokens | 失敗を確認した長さ |
|---|---:|---:|---|
| Choice、3選択肢 | 39.033B | 39.635B | 110〜128の全整数 |
| Noul、2選択肢 | 39.042B | 39.645B | 110、112 |
| Score、3選択肢 | 39.033B | 39.634B | 110、112 |

Choiceは96、104〜109でも成功した。Noul/Scoreは108/109/110/112だけを境界測定しており、
それ以外の長さを検証済みとは扱わない。109 tokensの残予算は約0.9%、108でも約2.4%にとどまる。

測定入力は自然文の品質評価用データではない。成功した長さだけから、すべての短い入力も
成功するという保証は導けない。上限付近は40Bに対して余裕が小さく、128 tokensには分割を使う。
境界測定JSON（ローカル生成物: `artifacts/int8_optimization_v2/final-update-limits.json`）には入力token列、
成功命令数、失敗時のIC0522、module/pack hashを保存している。

## 再現と検証

```sh
IC_LAYA_CANDLE=1 qrun -- bash tools/build_one.sh decision-engine
# 既存ローカルcanisterをupgrade後に、stableのモデルを復元する
.venv/bin/python tools/canister_infer.py --warmup --stepped \
  --input artifacts/laya-choice-128-input.json \
  --output artifacts/int8_optimization_v2/final-choice-128.json
.venv/bin/python tools/check_batch_inference_protocol.py \
  --evidence artifacts/int8_optimization_v2/final-choice-128.json \
  --output artifacts/int8_optimization_v2/final-protocol.json
```

- 実Wasmの整数参照比較760ケース、量子化の丸め・非有限値144ケースが成功。
- `laya-candle`とCandle有効の`decision-engine`のnativeテスト、Candle無効のcheckが成功。
- CLI/packのPythonテスト9件が成功。開始統合・後続バッチ・profile互換を含む。
- 実モデルのChoice 38、Noul 45、Score 37 tokensは、それぞれ13.338B／15.881B／12.959B命令で
  単一updateが成功。従来版と入力・pack・logitsが一致した（比較記録（ローカル生成物: `artifacts/int8_optimization_v2/final-parity.json`））。
- 最終WasmのSHA-256は `f0046c2a40a5f43de3c17f0c259535761e3b3d7362f0848bfcd1cf959efdef9d`。
  ローカルに配置したmodule hashとの一致を確認した。

測定対象はowner向けraw-token API。`evaluate`のReceipt発行やexecutor経路、
任意の文章・選択肢数、mainnetの成功保証ではない。量子化方式・モデル・tokenizerは変更していない。
