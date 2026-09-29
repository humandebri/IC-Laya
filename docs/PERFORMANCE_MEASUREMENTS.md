# ICP性能の実測

> Historical note: this document describes an earlier revision. Statements about missing implementations, tests, or measurements are not the current project status. Start with the [English local inference guide](GETTING_STARTED.md) and [current measurement index](README.md).

対象: `tools/measure_inference.py`（local replica上の実測）
測定日時の記録: `artifacts/inference_measurements.json`

**この文書は測定値と、明示的に区別した外挿値から成る。** 外挿を測定値として扱わない。

## 何を測ったか

`tools/generate_fixtures.py --tier` が生成する合成packを、`IC_LAYA_CANDLE=1` でビルドした decision-engine canister に
`begin_upload` → `upload_chunk` → `start_warmup` → `warmup_next` で投入し、
`register_schema` / `register_calibration` を通してから `evaluate` を呼び、
canister 自身が返す `measured_instructions` を読んだ。

合成weightは**同じ `laya-candle` カーネル**を通るため、層あたり・tokenあたりの計算量は意味を持つ。
一方で**判断品質や上流parityの証拠にはならない**。

評価は設計の返金workflowと同じ三primitive（Noul → Choice → Score）で、入力は Compact128 profile に収まる長さ。

## 測定値

| tier | hidden | encoder層 | params | pack | 質問あたり instructions | 実測 wall |
|---|---|---|---|---|---|---|
| `measure-s` | 128 | 2 | 約0.2M | 3.1 MiB | **193M / 207M / 228M** | 0.30 s |
| `measure-m` | 512 | 4 | 約13M | 81.1 MiB | **4.73B / 5.02B / 5.43B** | 0.93–1.02 s |

（3値は Noul / Choice / Score の順。候補数が増えるほど高い。）

正規化すると **encoder層 × token あたり**:

- `measure-s`: 3,270,454 instructions
- `measure-m`: 39,581,128 instructions

hidden 128 → 512（4倍）で **12.1倍**。指数は `log(12.1)/log(4) = 1.80` で、二次（2.0）よりやや緩やか。
これはattentionの `O(hidden²)` に加えて、層数に比例しない定数項（embedding lookup、scorer、Candid decode）が小さいtierでは相対的に重いためである。

## 外挿（測定値ではない）

実checkpointは **encoder 28層 / hidden 1024 / vocab 50368**。`measure-m` から hidden をさらに2倍に上げ、
28層・128 tokens に当てはめると:

| 前提 | 421Mモデルの1質問あたり | 20B目標に対して | 40B上限に対して |
|---|---|---|---|
| hiddenに対し二次（保守的・上限側） | **567B instructions** | 28倍超過 | **14倍超過** |
| 実測指数 1.80 を延長 | **494B instructions** | 25倍超過 | **12倍超過** |

**結論: 現状のF32実装は、ICPの1 updateあたり40B instructionsという公開上限に対して12〜14倍超過する。**
設計の受入目標20Bに対しては25〜28倍である。これは性能の「不足」ではなく、**アーキテクチャ選択の棄却**に近い。

外挿の仮定と限界:

- 実checkpointの `vocab_size` は 50368 で `measure-m` の 2048 より大きい。embedding lookup と最終normは層数に比例しないため外挿に含まれず、**実測はこれより悪化する方向**に働く。
- 実checkpointは `local_attention=128` のsliding windowを持つ。`measure-m` も `local_attention=128` に設定してあるが、`global_every` の層配置は層数に依存するため厳密ではない。
- decision head（2層）の分は含めていない。encoder 28層に比べて小さいが、無視できる大きさではない。
- したがって上表の値は**下限寄りの推定**であり、実際はこれより大きい可能性が高い。

## ICPのwasmはSIMDを実行できる（実測）

v2.0は「SIMD専用kernel」を将来の最適化として挙げていたが、**使えるかどうか自体が未確認**だった。
wasm-feature probe canisterを作り、local replicaへinstall・実行して確認した。

probeは `std::arch::wasm32` のintrinsicを使い、wasmがSIMD必須であることを検証したうえで
（`wasm-tools validate --features mvp` は失敗、`--features simd` は成功）実際に呼び出した:

| 命令 | 結果 | 意味 |
|---|---|---|
| `i32x4.mul` | 実行成功（50） | SIMD算術が動く |
| `i32x4.dot_i16x8` | 実行成功（36） | **4レーン整数ドットが動く** |
| `i16x8.extend_low/high_i8x16` | 実行成功（72） | i8→i16拡張が動く |
| `i16x8_dot_i8x16_i7x16_s` | **Rustでは未公開** | 直接のi8ドットはコンパイルできない |

したがって**INT8カーネルは「i8→i16拡張 + `i32x4.dot_i16x8`」の経路で書ける**（4 MAC/命令）。
これはADR-010の方針が実行可能であることの確認であり、同時にADR-017として記録した。

### ただし `+simd128` は今は有効にできない（candle-core 0.11.0 のブロッカー）

上のprobeが確認したのは「wasm SIMD命令をICが実行できる」ことであって、
**この依存グラフをSIMD有効でビルドできる**ことではない。実際にはできない。

`simd128` は wasm32-unknown-unknown の既定機能ではない（実測）:

```
$ rustc --target wasm32-unknown-unknown --print cfg | grep target_feature
bulk-memory / multivalue / mutable-globals / nontrapping-fptoint /
reference-types / sign-ext                    # simd128 は無い
```

`.cargo/config.toml` に `-C target-feature=+simd128` を足すと、**candle-core 0.11.0 が
コンパイルできない**（実測、`IC_LAYA_CANDLE=1 cargo build --target wasm32-unknown-unknown`）:

```
error[E0433]: cannot find type `CurrentCpuF16` in this scope   (x38, candle-core)
```

原因は上流の不整合である。`candle-core` の `src/cpu/mod.rs` は `vec_dot` 系ヘルパを
`any(target_feature = "neon", target_feature = "avx2", target_feature = "simd128")` で
ゲートしているのに、`src/cpu/simd128.rs` は `CurrentCpu` しか定義していない
（`avx` と `neon` は `CurrentCpuF16` も export している）。そのため
`.cargo/config.toml` ではこのflagを**意図的に**設定していない。

**matmulがこの理由でscalarになっているわけではない。** candleのf32 matmulは `gemm` crateを
通り、gemmのwasm32 SIMDマイクロカーネルはビルドに含まれ、`get_wasm_simd128()` の既定
（candleが有効化する `wasm-simd128-enable` により **true**）で選択される。
flagで失っているのはcandle自身のwasm SIMD（要素演算・正規化）だけで、内訳では数%のshareである。
したがって**このflagは第一のレバーではない**。

### gemmのSIMDカーネルは実際に効いている（実測A/B）

独立したbench canister（`gemm` を直接、candleと同じ引数・同じ形状
`m=128, n=4096, k=512` = measure-mの `wi` 射影で呼ぶ）で命令数を測った:

| gemmのカーネル | instructions | instructions/MAC |
|---|---|---|
| wasm32 simd128（`set_wasm_simd128(true)`） | 669,805,104 | **2.50** |
| scalar（`false`） | 2,912,265,769 | **10.85** |

**4.35倍**の差があり、`DEFAULT_WASM_SIMD128` も実行時の既定値も `true` である。
IC-Layaは既にSIMDカーネルの側にいる。フェーズ実測が3.9〜4.2 instr/MACなのは、
純粋なmatmul 2.50にnormと重み転置コピーの分が乗った値として整合する。

> **測定上の落とし穴（実際に踏んだ）**: gemmは最初の呼び出しで選んだカーネルを
> `static GEMM_PTR: AtomicPtr` に固定して以後再利用する。`set_wasm_simd128` を後から
> 呼んでも切り替わらないため、A/Bは**canisterを再installしてから**フラグを設定し、
> そのinstanceで最初のgemm呼び出しを行う必要がある。最初の計測はこれを踏んで
> 両方10.85になり、差が出なかった。

残る余地は「gemmのSIMDカーネル 2.50」から「f32x4の下限 0.5〜1.0」までの**約2.5〜5倍**で、
これはADR-017のINT8カーネル計画の担当領域である。

## フェーズ別の内訳（実測）

`measure_phases`（測定専用endpoint。cacheを迂回し、資金移動経路に触れない）で内訳を測った。

| tier | embedding | **encoder** | final_norm | decision | scorer | decode |
|---|---|---|---|---|---|---|
| measure-s | 0.0% | **70.3%** | 0.0% | 27.7% | 1.4% | 0.0% |
| measure-m | 0.0% | **83.3%** | 0.0% | 15.7% | 0.9% | 0.0% |

**encoderが支配的**で、softmax・norm・活性化・gather・decodeは合計1%未満である。
ADR-010の「hot linearへINT8/SIMDを適用する」という判断は、**この内訳で裏付けられた**。

## 測定の不整合（3度の誤り。訂正済み）

この文書は efficiency について**3度誤った**。いずれも測定ツール側の欠陥である。

### 誤り1: token長を128と仮定していた

`measure_phases.py` は理論MACを `tokens = 128` 固定で計算していた。しかし測定側が
渡していた `STATE` は短く、実際に render された入力は **27〜38 tokens** だった
（`inference_measurements.json` の `input_tokens` が 27 / 32 / 38 と記録している）。

この不整合により **instructions/MAC を約4倍過小評価**していた。
「encoder 1.43 instr/MAC」という記載は誤りである。

加えて、測定側は独自にリクエストを組み立てていたため、`generate_fixtures.py` が書いた
128-token の `input.json` を**一度も使っていなかった**。生成側と測定側で入力が
食い違っていたことになる。

### 誤り2: token長を記録していなかった

`measure_phases` は `input_tokens` を返さないため、**内訳測定がどの長さで行われたか
記録が無かった**。効率を主張する根拠が欠けていた。

### 修正

- `--full-profile` を追加。state を埋めて **実際に128 tokens** の入力を作る
- 理論MACは**実測した render 長**から計算する（仮定しない）
- `rendered_tokens` を必ず記録する
- 短い入力と128-token入力を**両方**測る（attentionはtoken数に対して二次、他は線形なので区別が必要）

### 誤り3: MLPのMACを過大に数えていた（measure-mの条件で1.242倍）

`theoretical_macs_encoder` は wi と wo の**両方**に `2*inter*h` を課していた。
`expected_tensors` の `wo.weight` は `[hidden, intermediate]` なので、1 token あたりは

- wi（gate + up）= `2*inter*h`
- wo = `inter*h`

で、MLPは `3*inter*h` が正しい。`encoder_macs` は1 tokenあたり
`4*h*h + 3*inter*h + 2*window*h` で計算する。

**ただし過大だったのはMLP項だけで、他の項は正しかった。** したがって分母全体の過大は
4/3（1.333）ではなく、MLP項が全体に占める割合（measure-mで約73%）のぶんだけ希釈されて
**1.242倍**になる。MLP項だけを4/3にすると全体は1.242倍、という関係である。

3つの誤りは**同じ向き**（分母を大きくする）に効いていた。つまり訂正前の数値はすべて
「実装はもっと効率的」に見える方向に歪んでいた。

## 訂正後の効率: カーネルはむしろ非効率である

128-token profile での実測（`--full-profile`、いずれも `rendered_tokens = [128,128,128]`）。
分母は誤り3の訂正後の `encoder_macs`:

| tier | 1質問合計 | 理論MAC（encoder） | instructions/MAC（全体） | encoder単独 |
|---|---|---|---|---|
| measure-m | 10,946,018,414 | 2,214,592,512 | 4.94 | 4.11 |
| measure-6l768 | 26,585,076,637 | 5,445,255,168 | 4.88 | 4.10 |

**f32x4 SIMD の下限は 0.5 instructions/MAC**（wasm SIMD に FMA が無いため `f32x4.mul` +
`f32x4.add` の2命令で4 MAC。gemm 自身の wasm カーネルも `add(c, mul(a,b))` で実装している）。
したがって**現状は下限の約10倍**である。INT8（`i32x4.dot_i16x8` は 4 MAC/命令）なら更に下がる。

したがって以前の結論:

> 実装効率はすでに下限近く（1.79〜1.99 instr/MAC）で、実装の作り直しでは速くならない

は**誤りである**。正しくは:

> カーネルは下限の約20倍非効率で、**実装を直す余地は大きい**。

原因の候補（未確定）: matmul が f32x4 を使い切れていない、`Tensor` の中間生成と
メモリ往復、per-op オーバーヘッド。**どれが支配的かはまだ測っていない。**

## 律速の最終分解: カーネルは良い、オーバーヘッドが悪い

token長を振って測った（`--token-sweep`、`measure-m`）。

| 質問 | instructions |
|---|---|
| T=32 | 5,018,315,905 |
| T=48 | 6,007,196,304 |
| T=64 | 6,967,632,832 |
| T=96 | 8,931,010,028 |
| T=128 | 10,949,465,117 |

`instr ≈ 3,119,969,913 + 59,088,767·T + 16,021.81·T²`（最大残差0.13%）。
**token長に依存しない固定項が3.12B**あり、これはT=32で全体の62%、T=128でも28%を占める。

### 成分ごとの効率（T=128）

| 成分 | 理論MAC | 実測instructions | instr/MAC | 判定 |
|---|---|---|---|---|
| layer.mlp_up | 1,073,741,824 | 1,043,795,407 | **0.97** | **最適**（理論下限0.25の4倍） |
| layer.mlp_down | 536,870,912 | 519,803,922 | **0.97** | **最適** |
| layer.attn | 603,979,776 | 633,610,271 | **1.05** | **最適** |
| layer.mlp_act（GeLU） | 2,097,152 | 66,081,838 | 31.51 | 非効率（ただし小さい） |
| **decision head** | 805,306,368 | 1,798,154,941 | **2.23** | **約2倍のオーバーヘッド** |

**encoderのmatmulは理論下限の約4倍で、f32としては妥当に最適化されている。**
つまり「行列積そのものを書き直す」余地は小さい（既に約1.0 instr/MAC）。
以前「下限の約20倍」と書いたのは、MACを**全フェーズ合計**で割っていたためである。

一方で:

- **decision headが2.23 instr/MAC**で、MAC比の2倍以上を消費している（約0.85Bの無駄）
- **固定項3.12B**の正体は未解明。T→0へ外挿した値であり、per-opオーバーヘッド、
  中間`Tensor`の生成、層ごとの固定費などが候補だが**特定していない**
- GeLUは31.5 instr/MACと非効率だが、絶対量は小さい

**したがって「効率の悪いところ」は実在する。ただし場所はmatmulではなく、
decision head と、token長に依存しない固定費である。** この2つで約4B/質問（T=128の36%）を占める。

## 目標構成の実測: 6層 / hidden 768

蒸留候補の構成（encoder 6層 / hidden 768 / 16 heads / intermediate 1968 / sliding 128）を
実測した。実checkpointと同じhidden幅・head数・窓・intermediate比を持ち、層数とvocabだけを落とした合成pack
（53.0M params、202 MiB）。

短いstate（27〜38 tokens）と、128-token profile の**両方**を測った。

| 入力長 | 質問 | instructions |
|---|---|---|
| 短い state | Noul / Choice / Score | 11.59B / 12.27B / 13.28B |
| **128 tokens** | Noul / Choice / Score | **26.59B / 26.59B / 26.59B** |

**128-token profile では3質問合計 79.8B instructions。** これは:

- ICPのupdate call上限40Bに対して **2.0倍超過**
- 設計目標20Bに対して 4.0倍
- 壁時計（2B/秒）で **約40秒**

以前この節に「36.2B、上限の0.90倍で収まる」と書いたが、**それは短い入力での値だった**。
設計が想定する128 tokensでは**上限を超える**。

サブフェーズの構成は4層h512のときとほぼ同じ（mlp_up 36%、attn 29%、mlp_down 18%、decision 16%）。
**hiddenを広げても律速は変わらない**ため、INT8の対象は同じでよい。

### INT8を当てた場合の見通し

MLP（54%）とattention（29%）がINT8の対象で、合計83%。**×3** を当てると:

| 構成 | 3質問合計 | 壁時計 |
|---|---|---|
| 6層 h768（F32、128 tokens） | 79.8B | 約40秒 |
| + カーネル改善×5（後述） | 約16B | 約8秒 |
| + INT8×3 併用 | 約5B | 約2.7秒 |

**つまり「6層 h768 + INT8×3」で約6秒**、設計目標20B（10秒）を満たす。
encoder共有が効けばさらに3.7秒程度。

**注意**: この表のINT8行はまだ**外挿**である。実測は「6層 h768 F32 = 36.2B」まで。
INT8カーネルの実装と再測定が次の作業である。

## 層数を増やす方向の候補は、コストの観点で逆効果

`models/verdict-151m`（GLiClassModel、encoderはModernBERT-base）が置かれている。
Jev系のオープン実装であり、校正temperatureを持つ点は参考になる。ただし**この問題に対しては
逆方向**である:

| | 6層h768（適合候補） | verdict-151m |
|---|---|---|
| encoder層 | **6** | **22** |
| hidden | 768 | 768 |
| intermediate | 1968 | 1152 |
| vocab | 4,096 | **50,370** |
| params | 53M | 151M |

層数が3.7倍、vocabが12倍なので、MLP項がやや小さいことを差し引いてもコストは数倍になる。
6層h768で既に128-token profileは上限40Bの2.0倍なので、**22層では届かない**。

また head の形も異なる: verdictは `class_token_index` と `scope: restricted_5_candidate_selection` を
持つ**単一クラストークン方式**で、5候補の選択に特化している。一方この設計が必要とするのは
Choice/Noul/**Score（尺度付き順序）**の3primitiveで、Scoreは順序尺度と分布を要求する。
加えて**校正はschema単位**である必要がある（`Calibration` は `schema` に束縛される）。

したがって verdict-151m は「backboneの候補」ではなく、**別の設計の既成モデル**として扱う。
採用するなら、三primitiveと尺度を満たすヘッドを改めて学習することになり、蒸留と同じ評価負担になる。

## 壁時計時間: ここが本質的な制約

公式ドキュメント（[Resource limits](https://docs.internetcomputer.org/references/resource-limits/)）:

| 上限 | 値 |
|---|---|
| per **update** call | **40 billion** instructions |
| per **query** call | 5 billion instructions |
| per install / upgrade | 300 billion instructions |
| 目標スループット | **2 billion wasm instructions / thread / 秒** |

**この2B/秒が効く。** instructionsは時間に比例する:

| 構成 | instructions | 壁時計 |
|---|---|---|
| 現状 421M（投影） | 493B | **約247秒** |
| + INT8 ×4 | 127B | 約64秒 |
| 蒸留 8層 h768 + INT8×3 | 28B | 約14秒 |
| 蒸留 6層 h640 + INT8×3 | 15B | 約7.5秒 |
| （上限）update call | 40B | 20秒 |
| （目標）設計値 | 20B | 10秒 |

**目標から逆算すると:**

| 応答時間 | 必要なinstructions |
|---|---|
| 1秒以内 | 2B以下 |
| 3秒以内 | 6B以下 |
| 10秒以内 | 20B以下 |
| 30秒以内 | 60B以下 |

**注意**: この2B/秒という値は**実測ではなく公式の目標値**である。実測したlocal replicaは
cycle課金が無く、この数値と一致する保証はない。桁の見当をつけるために使っている。

## 示唆

- **律速は命令スループットだが、実装には大きな余地が残っている。** instr/MAC は
  約4.8〜5.2（encoder）で、f32x4 の理論下限0.25の**約20倍**である。以前「下限近く」と
  書いたのは token 長の誤りの産物だった。**カーネル改善とMAC数削減の両方が効く。**
- **MAC数削減（モデルを小さくする）は依然として支配的**である。6層h768でさえ
  128-token profile では上限を超える。
- **この構成のままでは実checkpointを載せられない。** F32での正しさを優先した設計判断（ADRの順序: F32で一致確認 → buffer再利用 → INT8/SIMD → 蒸留）は妥当だったが、その最初の実測で予算に届かないことが確定した。
- 必要な削減は**最低でも12倍**（40B上限に収めるだけでも）。INT8化で見込める4倍では届かない。
  残差は**3.1倍**（123B → 40B）で、これはADR-017のとおり融合カーネルで中間tensor生成を削ることで
  埋める見込みである。20B目標に対しては6.2倍。
- **INT8 + SIMDだけでは届かない。** 実測の効率はすでに 1.43 instructions/MAC で、
  INT8 SIMDの理論上限（4 MAC/命令）を当てても**最大で3〜4倍**が現実的な天井である。それを当てると:

| INT8/SIMDの伸び | 421M投影 | 40B比 | 20B比 |
|---|---|---|---|
| ×2 | 249B | 6.2倍 | 12.5倍 |
| ×4（理論上限近く） | 127B | **3.2倍** | 6.4倍 |
| ×6（非現実的） | 86B | 2.2倍 | 4.3倍 |

  **×4でも40Bの3.2倍超過**である。したがって:

  - INT8/SIMDは必要だが**十分ではない**。単独では本番採用基準を満たせない。
  - **蒸留が必須**である。8層/hidden 768 まで落とせば 84B（40Bの2.1倍）、
    6層/hidden 640 で 45B（1.1倍）。INT8/SIMD（×3〜4）と組み合わせれば
    **6〜8層で20B以内**に入る。
  - つまり正しい順序は「**蒸留で規模を落とし、その上でINT8/SIMD**」である。
    INT8だけを先に完成させても本番基準には届かない。
- 「Scoreを削る」「尺度説明を短縮する」「入力を切る」といった意味を削る最適化は、この測定結果を理由にしても**依然として認められない**。削るなら計算側（量子化・蒸留・カーネル）である。

## 未測定（重要）

1. **heap使用量**。canisterのheapを直接読む口がないため、`warm heap 2.5GiB以下` / `cold peak 3.0GiB以下` は**測っていない**。
   実checkpointのF32 packは1.57 GiBで、warm-upは1 tensorずつstableからheapへ読む。`Builder::push` が `Tensor::from_raw_buffer` でstableの読み出し元bufferをそのまま使うため、恒久的な二重化は避けられる設計だが、**実測していない**。
2. **1.57 GiB packの投入そのもの**。今回の最大は81 MiBである。
   CLI経由の `upload_chunk` は `icp canister call` が引数をテキストで受け取りCandidが各バイトを `\xx` に展開するため、argv長で1 MiB chunkが使えない（実測: `OSError: Argument list too long`）。256 KiB chunkで1.57 GiBは約6400回の呼び出しになり、この経路では非現実的である。
   これは**canister側の制約ではなく、CLIというクライアントの制約**である。実運用のアップロード経路（binaryを渡せるクライアント）を別途設計する必要がある。
3. **wall-clockの本番相当値**。local replicaは cycle 課金されないため、この 0.3〜1.0 秒を mainnet の見積もりに使ってはならない。判定に使ったのは `instructions` のみである。

## 再現手順

```bash
python tools/generate_fixtures.py --tier measure-s --tier measure-m
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine
python tools/measure_inference.py --tier measure-s --tier measure-m --chunk-kib 256 --timeout 900
```

`--chunk-kib` はCLIの引数長制約に対する回避策であり、canisterの上限（`upload_chunk` は1 MiBまで）とは別物である。

## 付録: gemmのカーネルA/B（この文書の2.50 / 10.85の出所）

`gemm` を直接呼ぶ最小のcanisterを別途立て、candleと同じ引数・同じ形状で命令数を測った。
依存は `gemm = { version = "0.19.0", features = ["wasm-simd128-enable"] }` と `ic-cdk` だけ。
（実際に使ったcanisterは下と同じ呼び出しにchecksumと`instructions/MAC`を足して返す。）

```rust
#[ic_cdk::update]
fn bench(m: u32, n: u32, k: u32, iters: u32, simd: bool) -> (u64, u64) {
    gemm::set_wasm_simd128(simd);
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let lhs = vec![1.0f32; m * k];
    let rhs = vec![1.0f32; k * n];
    let mut dst = vec![0.0f32; m * n];
    let start = ic_cdk::api::instruction_counter();
    for _ in 0..iters.max(1) {
        unsafe {
            gemm::gemm(m, n, k,
                       dst.as_mut_ptr(), 1, n as isize, false,
                       lhs.as_ptr(), 1, k as isize,
                       rhs.as_ptr(), 1, n as isize,
                       0.0, 1.0, false, false, false,
                       gemm::Parallelism::None);
        }
    }
    (ic_cdk::api::instruction_counter() - start, m as u64 * n as u64 * k as u64)
}
```

**フラグはcanisterの再install後に、そのinstanceで最初のgemm呼び出しを行う前に設定する。**
gemmは最初の呼び出しで選んだカーネルを `static GEMM_PTR: AtomicPtr` に固定するため、
後から `set_wasm_simd128` を呼んでも切り替わらない（最初の計測はこれを踏んで両方10.85になった）:

```bash
icp canister install <bench> -e local -y -m reinstall --wasm bench.wasm --args '()'
icp canister call <bench> set_flag '(true)' -e local --candid bench.did
icp canister call <bench> bench '(128 : nat32, 4096 : nat32, 512 : nat32, 1 : nat32, true)' \
  -e local --candid bench.did
```

`m=128, n=4096, k=512` は measure-m の `wi`（`[2*inter, h]` = `[4096, 512]`）と128-token入力を
そのまま写したもので、candleの `MatMul::f`（`candle-core/src/cpu_backend/mod.rs`）が渡す
引数・stride・alpha/beta をそのまま使っている。
