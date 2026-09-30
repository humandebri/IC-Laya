# int8推論の詳細計測と改善

> 最新の実測は[F32書戻し最適化](INT8_F32_WRITEBACK.md)を参照。

> この文書は4×4タイル版の履歴。最新の16×16・SIMD量子化・2 updateの結果と単一update境界は
> [INT8_OPTIMIZATION_V2.md](INT8_OPTIMIZATION_V2.md)を参照。以下の88-token境界と3 updateは旧版の値。

2026-09-22、ローカルcanister `4caro-hl777-77775-aaaba-cai`。
固定revisionの実Laya、同じint8 pack、同じ128-token Choiceを比較する。
量子化方式・重み・入力・層数を変えず、整数行列積の実装を改善した。

## updateで扱える入力長

2026-09-22〜23（JST）に計測した実LayaのW8A8 packについて、確認済み範囲は次のとおり。
ここでtokensは本文だけでなく、質問・選択肢・特殊tokenを含む入力全体を指す。
1Bは10億命令。ICの単一update上限40Bとの比較は
[公式resource limits](https://docs.internetcomputer.org/references/resource-limits)に基づく。

| 実行方式 | 入力 | 実測命令数 | 確認結果 |
|---|---|---:|---|
| `infer_tokens`の単一update | 45 tokens、Noul 2選択肢 | 20.586B | 完走。40Bに対して約48.5%の残予算 |
| 最大16ステップずつの分割update | 128 tokens、Choice 3選択肢 | 合計59.420B、最大batch 31.619B | 完走。開始1回＋推論2回の計3 update |
| 128 tokensの単一update | 上記Choice入力 | 分割時の合計が40Bを超過 | 一括実行可能とは扱わない。分割を使用 |

### 単一updateの境界測定（2026-09-23 JST）

Choiceの本文tokenを反復・短縮し、質問・3選択肢・終端SEPを固定した長さ別測定を追加した。
**最大成功実測は88 tokens。ただし86/87は失敗するため、88を一律の受付上限にはできない。**

| Choiceの総tokens | 単一updateの命令数／結果 |
|---|---:|
| 80 | 36.110B、成功 |
| 81 | 37.265B、成功 |
| 82 | 38.374B、成功 |
| 83 | 39.486B、成功 |
| 84 | 37.998B、成功 |
| 85 | 39.161B、成功 |
| 86、87 | 40B超過（IC0522） |
| 88 | 39.910B、成功。残予算約0.23% |
| 89〜92、96 | 40B超過（IC0522） |

Noul（2選択肢）とScore（3選択肢）も85/88で成功し、86/87/89で40Bを超過した。
88-tokenはそれぞれ39.918B / 39.912B命令。3系列とも最大成功実測は88だが、
残予算は約0.2%にとどまる。

4-tokenタイルと端数処理の違いで命令数は単調増加しない。85までの測定成功も、
任意の文章・選択肢数・APIの成功保証ではない。93〜95と97〜127はこの境界測定では未測定。
長さ上限だけで成功を保証するには、選択肢数・タイル端数を含む追加検証と余裕が必要。
根拠は 境界測定JSON（ローカル生成物: `artifacts/int8_update_limits.json`）。各入力token列と成功命令数、
失敗時のIC0522、module/bundle hashを保存し、失敗を成功や推定値で補っていない。

```sh
.venv/bin/python tools/measure_update_limits.py --cases choice --lengths 80,81,82,83,84,85,86,87,88,89,90,91,92,96
.venv/bin/python tools/measure_update_limits.py --cases noul,score --lengths 85,86,87,88,89
```

### 分割回数について

1層ずつ32ステップの旧実装を、最大16ステップずつ進めるバッチAPIへ変更した。
2026-09-23 JST、同じ実モデル・128-token Choiceで次を確認した。

| 呼出し | 処理 | 命令数 |
|---|---|---:|
| start | embeddingと初期化 | 0.019B |
| batch 1 | ステップ1〜16 | 31.619B |
| batch 2 | ステップ17〜32 | 27.782B |
| 合計 | **計3 update（従来33回）** | **59.420B** |

各推論updateが40B以内で完走し、従来版と入力・pack hashおよびlogitsが完全一致した。
最大バッチの残予算は約21%。ローカルwall timeは11.03秒（CLIを含む、mainnetの見積もりではない）。
新Wasmのmodule hashは`22514c506ad4a4b2120fc055fb40e633dcdbe6d0462adef6688503baf30276d7`。
命令数の差をバッチ化だけの効果とは扱わない。主目的はupdate回数の削減。
証跡: [128-tokenバッチ推論](../artifacts/int8_batched_choice-128.json)。

`--stepped`は既定で16ステップずつ。`--steps-per-call 1`で従来方式、
`--profile`は常に1ステップずつ計測する。異なるモデルでは16ステップが40Bに収まる保証はなく、
必要に応じて`--steps-per-call`を小さくする。この測定時点では開始処理を統合していない。後続のV2で計2 updateのAPIを実装・検証した。

```sh
.venv/bin/python tools/canister_infer.py --stepped --input artifacts/laya-choice-128-input.json --output artifacts/int8_batched_choice-128.json
.venv/bin/python tools/check_batch_inference_protocol.py
```

分割updateは**現実装の入力上限128 tokensまで、上表の1入力で完走を確認**した。
全入力・全選択肢数での完走保証ではない。`MAX_TOKENS=128`を適用するため、129以上は
分割しても受け付けない。128を超える拡張には別途実装・検証が必要。

いずれもowner向けraw-token APIの測定であり、`evaluate`のReceipt発行やexecutor経路の
上限を示すものではない。モデルはwarm済みで、アップロード・warmupの費用は含まない。
単一updateの境界測定は旧Wasm（module hash `8b7106bdb1d54194c1c0431905dffaf93098ad6fac4e2c50137817f662b32490`）
によるローカル測定で、バッチ版Wasmでの
単一update境界の再測定は未実施。上記3 updateの結果は新Wasmで確認した。mainnet実測は未実施。

証跡: 45-token単一update（ローカル生成物: `artifacts/int8_tiled_noul_single.json`）、
[128-token分割update](../artifacts/int8_batched_choice-128.json)。

## 改善前の内訳

`artifacts/int8_profile_baseline_128.json` が新たに取得した詳細計測。
以前の138.229Bは計測なしの実装、今回の138.964Bは詳細計測と
バッチ単位のactivation準備を含む比較用baselineなので混同しない。

| 区間 | 命令数 | 全体の割合 |
|---|---:|---:|
| int8行列積・出力scale適用 | 125.615B | 90.4% |
| activation量子化 | 3.875B | 2.8% |
| attention QK積 | 2.240B | 1.6% |
| softmax | 1.722B | 1.2% |
| MLP GeLUとgate | 1.573B | 1.1% |
| attention AV積 | 1.446B | 1.0% |
| RoPE | 0.980B | 0.7% |
| LayerNorm | 0.831B | 0.6% |
| 入力コピー | 0.093B | 0.07% |

int8行列積の中ではencoder MLP up `[128,5248,1024]` が51.819B、
QKV射影 `[128,3072,1024]` が32.526B、MLP down `[128,1024,2624]` が24.803B。
shapeは `[tokens, output_features, input_features]`。
主要な問題はembeddingの大きさやモデルロードではなく、これらの行列積だった。

生成Wasmも確認した。以前は各出力要素につき`dot`を呼び、16要素ごとのループで
入力とweightを毎回ロードし、両方のlow/highをint16へ拡張していた。
同じ入力がoutput数だけ、同じweightがtoken数だけ繰り返し処理されていた。

## 変更

- 4 tokens × 4 output rowsの16個の内積を一緒に計算する。
- 各16要素ブロックのweight/inputを読み込み・拡張し、それぞれ4個の内積で共有する。
- 16個のint32 SIMD accumulatorを使い、loop制御も共有する。
- 行ポインタをループの外で用意する。整数和の順序は変わるが、K≤16384の範囲で
  `16384*128*128 < i32::MAX` なので整数演算は厳密に等価。
- token/output数が4の倍数でない末尾、Kが16の倍数でない末尾も処理する。
- activationのround/scale、出力へのF32 scale適用順、attentionなどの浮動小数点演算は保持する。

## 再計測

```bash
IC_LAYA_CANDLE=1 qrun -- bash tools/build_one.sh decision-engine
# ローカル試験canisterをupgrade後、stableのモデルを復元して計測
.venv/bin/python tools/canister_infer.py --warmup --profile \
  --input artifacts/laya-choice-128-input.json \
  --output artifacts/int8_profile_tiled_128.json
.venv/bin/python tools/summarize_int8_profile.py \
  artifacts/int8_profile_baseline_128.json artifacts/int8_profile_tiled_128.json \
  --out artifacts/int8_profile_comparison_128.json
```

`profile_token_step`はowner専用で、通常の継続推論と同じstepを進めながら命令数を収集する。
返すshapeと区間名でlayerごとの支配項を追える。現在の計測点は重複しない区間に置いている。
各stepの合計には区間外のreshape/residual、計測bookkeepingなども含まれる。
wall timeはローカル環境＋CLI通信を含み、mainnetの速度見積もりではない。

## 128-token実測結果

| 指標 | 改善前 | 4×4タイル化後 |
|---|---:|---:|
| 全体 | 138.964B | **59.747B** |
| int8行列積 | 125.615B | **46.397B** |
| 行列積以外 | 13.349B | 13.351B |

**全体57.0%削減（命令数で2.33倍）、行列積63.1%削減（2.71倍）**。
同一input SHA-256・bundle SHA-256を確認し、最終logitsはserialized F32で完全一致した。
各phaseの比較は `artifacts/int8_profile_comparison_128.json` に保存。
差はほぼ行列積に集中しており、浮動小数点演算やモデルを変えて得た改善ではない。
計測shapeから数えた線形層のMACは両方47,147,125,760回。
行列積のinstructions/MACは **2.664 → 0.984** になった。

wall timeもartifactに記録するが、baseline採取中にはnative/Wasmビルドが並行しており、
CPU負荷を統制していない。**77.3秒→27.2秒を速度倍率の根拠にしない**。
比較の主指標はcanister自身のinstruction counter。

## 残るボトルネックと限界

- 改善後も行列積が77.7%。次の候補はタイル寸法・SIMDレジスタ使用量の比較。
  より大きいタイルは命令を共有できる一方、レジスタからの退避でwall timeが悪化し得るため、
  今回の結果だけでは採用しない。
- 次は量子化6.5%、QK積3.8%、softmax2.9%、GeLU/gate2.6%、AV積2.4%。
  量子化の除算を逆数乗算に変えると境界の丸め結果が変わり得るため、今回は維持した。
- 約59.7Bは1 updateの40Bを依然超える。継続推論が必要で、単一update化は達成していない。
- 精度や校正の検証範囲は前回と同じ。今回確認したのは最適化前後の数値不変性。

## 最終Wasmの通常推論（計測spanなし）

| 入力 | 変更前 | 変更後 | 削減 |
|---|---:|---:|---:|
| choice-128 | 138.229B | 59.733B | 56.8% |
| noul | 47.658B | 20.729B | 56.5% |
| choice | 40.221B | 18.208B | 54.7% |
| score | 39.156B | 17.111B | 56.3% |

4入力すべてで変更前のcanister出力とF32 logitsが完全一致。
128-tokenの最大1ステップは1.999B命令。
`artifacts/int8_tiled_validation.json` と各 `int8_tiled_*.json` に証跡を保存。

検証: Rust workspace（candle有効）、candle無効check、Wasm build、Python 56 tests。
タイル境界・端数・signed値・K=16384の整数参照比較、計測がlogitsを変えないこともテストした。

さらに、初期実装で40B上限を超えた45-token Noulが、最終版では**単一updateで20.586B**、
ローカルwall time 2.71秒で完了した。継続版とlogitsも完全一致。
証跡: `artifacts/int8_tiled_noul_single.json`。全長で一括実行可能になったという意味ではない。
再送・不正step・不正job・不正入力・非owner・計測APIの権限検査も通過。

## レビュー後のWasmカーネル回帰検査

nativeの境界テストはscalar実装を通るため、別途、実際のWasm版`Int8Matrix::forward`を
独立したi64内積と比較する検査を追加した。160ケースでタイル・端数、正負の最大値、
ゼロ、K=1/15/16/17/31/32/33/1024/2624/16384を検証し、Node上で全件一致した。
CIでもcanisterビルドに加えてこの検査を必須実行する。

```sh
cargo build --release --target wasm32-unknown-unknown -p laya-candle --example int8-wasm-check --locked
node tools/check_int8_wasm.mjs target/wasm32-unknown-unknown/release/examples/int8_wasm_check.wasm
```

`CARGO_TARGET_DIR`を設定している場合は、Nodeへ渡すWasmのパスも変更する。
この検査はカーネルの数値検証であり、IC上の命令数を再計測したものではない。
