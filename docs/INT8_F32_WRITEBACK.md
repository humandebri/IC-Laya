# INT8行列積のF32書戻し計測と改善

2026-09-23、既存ローカルcanister `4caro-hl777-77775-aaaba-cai` で、
同じW8A8 packと128-token Choice入力を比較した。従来の**46.632B → 45.578B命令**に削減し、
logits `[1.4615486, -2.2996664, -0.065881155]` は完全一致した。呼出しは2 updateのまま。
Bは10億命令。module hashは候補ごとの証跡JSONに記録した。

## F32が残る理由

各INT8線形層は、入力をtoken行ごとに動的量子化し、INT8積和をi32で保持する。
その後、整数和にactivation scaleと重みの行scaleを順に掛けてF32を返す。
次の線形層でも量子化するが、このモデルでは間にRoPE・attention・残差加算・
LayerNorm・GeLUまたはgateが入る。線形層同士を単純につなぐ箇所はない。
中間F32を全面的に排除するには、これらの演算と量子化方式も変更・検証する必要がある。

以前の128-token層別測定では、次の線形層へのF32入力コピーは全体の0.2%、
再量子化は1.7%。INT8行列積の78.3%には、整数積和に加えF32書戻しも含まれていた。
全線形層の出力は累計3,952万要素、約158 MBのF32書込みに相当する。

## 区間を分けた計測

同じ128-token合成入力と重みで、主要3形状のINT8積和とF32変換・scale適用・
出力書込みを、タイルごとに独立して計測した。scalar版の結果は以下のとおり。

| 形状 `[出力,入力]` | 計測用行列積全体 | 整数積和 | F32書戻し | 書戻しの比率 |
|---|---:|---:|---:|---:|
| `[3072,1024]` | 328.791M | 302.008M | 26.064M | 7.9% |
| `[5248,1024]` | 561.684M | 515.931M | 44.527M | 7.9% |
| `[1024,2624]` | 255.722M | 246.794M | 8.688M | 3.4% |

1Mは100万命令。区間計測はタイルごとにcounterを呼ぶため、コンパイラが生成するコードと
全体命令数が通常の行列積から変わる。**上表の比率は推定値**であり、通常実行の厳密な内訳とは扱わない。
一方、各形状の出力checksumが通常の`benchmark_int8_kernel`と一致することを確認した。
scalar版証跡（ローカル生成物: `artifacts/int8_optimization_v2/f32-components-scalar.json`）。

## 採用した変更

整数積和の結果をF32へ変換し、activation scaleとweight scaleを掛ける書戻しを、
Wasm `simd128`の4要素単位へ変更した。2回のF32乗算の順序は保持する。
1〜3要素の端数は従来のscalar処理を使う。INT8重み、動的量子化、
attentionなど他のF32演算は変更していない。

| 128-token合成shape | scalar書戻し | SIMD書戻し | 削減率 |
|---|---:|---:|---:|
| `[3072,1024]` | 322.377M | 311.896M | 3.25% |
| `[5248,1024]` | 549.088M | 531.185M | 3.26% |
| `[1024,2624]` | 266.403M | 262.910M | 1.31% |

表は量子化も含む`Int8Matrix::forward`全体で、3形状ともchecksumが完全一致。
SIMD版証跡（ローカル生成物: `artifacts/int8_optimization_v2/f32-components-simd.json`）。
実モデルの2 updateは1回目24.857B→24.291B、2回目21.774B→21.287B、
合計46.632B→45.578B（**追加2.26%削減**）。入力・pack SHA-256とlogitsも一致した。
実モデル証跡（ローカル生成物: `artifacts/int8_optimization_v2/final-f32-simd-128.json`）。

32ステップの層別計測でも全体46.922B→45.851Bとなり、`int8.matmul`区間は
36.723B→35.667B。削減分のほぼ全てがこの区間にある。
区間比較（ローカル生成物: `artifacts/int8_optimization_v2/f32-writeback-profile-comparison.json`）。

Choice入力の本文token数を変えた単一updateは109〜113 tokensで成功し、
114〜128 tokensはすべて命令上限で失敗した。**最大成功実測は113 tokens**で
39.988B命令、40Bまで約0.012Bしか残らない。この長さを一律の受付上限にはできない。
Noul/Scoreや異なる文面での113-token成功は未検証。
境界測定（ローカル生成物: `artifacts/int8_optimization_v2/f32-writeback-update-limits.json`）。

## 検証と限界

実Wasmで、従来の整数参照760ケース、量子化境界144ケースに加え、
非2冪のscaleと小数入力を使ったF32書戻し216ケースをbit単位で照合した。
Candle有効のnativeテスト、Candle無効のcheckと、2 updateの実モデル推論も成功した。
検証記録（ローカル生成物: `artifacts/int8_optimization_v2/f32-writeback-validation.json`）。

ローカルcanisterの実行用cyclesが測定途中で凍結閾値に近づいたため、
試験identityから同canisterへローカルcyclesを1T補充して未完了分を再開した。
cycles-ledgerのcanister口座へのtransferではない。

残る費用の中心は整数積和。F32書戻しを全て消してもこの区間を丸ごと削減できるわけではない。
全面的な整数化は、現在のlogits一致を維持できる単純な置換ではない。
ローカルcanisterだけの測定であり、mainnetのwall timeや任意入力の成功を保証しない。
