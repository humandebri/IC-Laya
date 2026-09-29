# 実Laya checkpointとの接続: 調査結果

> Historical note: this document describes an earlier revision. Statements about missing implementations, tests, or measurements are not the current project status. Start with the [English local inference guide](GETTING_STARTED.md) and [current measurement index](README.md).

対象: `convaiinnovations/laya-typed-decisions` @ `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` (immutable revision)

**この文書は構造の突き合わせ結果であり、数値parityの証明ではない。** logits比較は未実施。

調査は`tools/laya_port_bridge.py discover`で再現できる。safetensorsの**header 21 KiBのみ**を取得し、803 MiBのweightは落とさない。出力は`checkpoints/laya-port/`（gitignore対象）に`canonical.config.json`・`tensor_map.draft.json`・`port_report.json`として出る。

## 判明した事実

| 項目 | 値 |
|---|---|
| 構成 | `ModernBertForMaskedLM` encoder + 2層 decision head + option scorer + act head |
| パラメータ数 | **421.3M**（納品時の「421M」と一致） |
| 保存dtype | **F16のみ**（loader は F32 を要求） |
| F32 packサイズ | **1,685,175,320 bytes (1.57 GiB)** / canonical上限2 GiB |
| encoder | hidden 1024, 28層, heads 16, intermediate 2624, local_attention 128 |
| rope | full 160000 / sliding 10000, `global_attn_every_n_layers=3` |
| decision head | `head_layers=2`, FF 4096, GELU |
| special tokens | `[CLS]`=50281 `[SEP]`=50282 `[PAD]`=50283 **`[MASK]`=50284** |
| tokenizer | BPE vocab 50280 + added tokens、`tokenizer/tokenizer.json` 3.4 MiB |
| 学習 | 7313 updates, 1 epoch, 1.96 hours, `fine_tuned_from_checkpoint=true` |

## 想定と実物の差（すべて mapping に影響する）

`tools/pack_checkpoint.py`が想定していたcanonical名は、**`scorer.*`以外ほぼ一致しなかった**。未確認の対応表をそのまま埋め込んでいたら、loaderは起動時に落ちていた。

| 観点 | canonical の想定 | 実物 |
|---|---|---|
| 埋め込み | `embeddings.weight` | `encoder.embeddings.tok_embeddings.weight` |
| encoder注意 | `encoder.{i}.qkv.weight` / `.out.weight` | `encoder.layers.{i}.attn.Wqkv.weight` / `.attn.Wo.weight` |
| encoder MLP | `encoder.{i}.wi.weight` | `encoder.layers.{i}.mlp.Wi.weight` |
| decision head | `decision.{i}.qkv.weight` | `head.layers.{i}.self_attn.in_proj_weight` |
| qtype埋め込み | `qtype.weight` | `type_emb.weight` |
| scorer | `scorer.norm/dense/out` | `scorer.0/1/3`（2と4はDropoutで欠番） |
| dtype | F32 | F16 |
| mask token | fixtureは50283 | **50284**（50283は`[PAD]`） |

`type_emb.weight`が`[3, 1024]`であることは、**三primitive構成が実checkpointに実在する**ことを示す。`attention_bias=false`、`norm_bias=false`、dropout 0.0 はcanonicalの前提と一致した。layer 0のみ`attn_norm`が無いのは`first_layer_attention_norm=false`と整合する。

## 構造の突き合わせ結果

`tensor_map.draft.json`（canonical名 → 上流名）を作り、全canonical tensorについて上流のshapeと照合した。

- canonical tensorで mapping に無いもの: **0件**
- shape不一致: **0件**

つまり**名前とshapeのレベルでは完全に接続できる**。ただしこれは「重みの使われ方が同じ」ことを意味しない。

## canonical packに入らない上流 tensor（5件）

| tensor | 意味 | 扱い |
|---|---|---|
| `act_head.0.weight/bias`, `act_head.2.weight/bias` | escalate 判断の head | 設計は「act/escalate headも消さない。option logitsを計算した後の別枝として省略する」としている。loaderはまだ読まない。**削除ではなく未実装**として残す |
| `temperature` `[3]` F16 | primitive別 temperature = `[1.0148, 1.0374, 1.0575]` | 下記の仕様ギャップ |

`rl_agent_config.json`には`temperature_by_options`もある: `choice:2`=1.906, `choice:3-5`=1.760, `choice:6-10`=1.000, `choice:11+`=0.101, `noul:2`=1.983, `score:3-5`=1.251。

**仕様ギャップ:** Candidの`Calibration.temperature`は`f64`スカラー1個である。上流の校正は「primitive別」および「候補数別」に分かれている。スカラー1個に潰すと、候補数2のChoiceと11+のChoiceに同じ温度を当てることになり、校正の意味が変わる。`temperature_by_options`は`choice:11+`を含み、これは現行のChoice上限5候補の外側なので、全範囲を再現するには候補数上限の見直しも要る。**これは未解決の設計判断であり、黙ってスカラーに潰してはいけない。**

## 未解決の問い（parityの前に潰す必要がある）

1. `Wqkv` / `in_proj_weight`の行順。canonical実装はPyTorch流のQ,K,V順を仮定している。上流logitsとの一致で確認するまで未確定。
2. RoPEの適用位置とlayout（head分割の前か後か）。
3. sliding windowの意味。上流`local_attention=128`に対しcanonicalは`local_attention/2`を窓としている。片方が「窓幅」で他方が「片側距離」なら挙動が違う。
4. decision headがどの行を読むか（CLS行か、pooled行か、mask marker行か）。
5. `head_max_len=256`と`max_prefixes=6`が、Compact128（合計128 tokens、prefix 64）と整合するか。**上流は最大6候補を想定しており、canonicalのChoice上限5より広い。**
6. 入力promptの組み立て。canonicalの`render()`は`"{primitive} question: {instructions}"`という自作形式であり、上流Layaの学習時promptと一致する保証はない。**ここが違えば、名前とshapeが全部合っていてもlogitsは別物になる。**

## ICP受入目標への影響

F32 packが1.57 GiBなので、weightだけでwasm32 heap 4 GiBの約39%を使う。ただし`cold peak 3.0GiB以下`という目標に対しては、pack 1.57 GiB + 推論中の活性化 + stableからの復元bufferが同時に載る瞬間があり、余裕は1.43 GiBしかない。28層・hidden 1024・128 tokensのF32活性化と、1 tensor最大256 MiBのwarm-up一時bufferを考えると3 GiBは厳しい。F16保持かINT8化かは、parityを取った後の実測で決めるべきで、先に量子化して意味を削るのは設計順序（F32で一致確認 → buffer再利用 → 実測で重い演算をINT8/SIMD化）に反する。

## 次にやること

1. `--weights`で実際に803 MiBを取得し`export`を実行、F32 packを作る（この環境では未実施）。
2. 上流実装を読み、上記6つの未解決の問いを確定させて`tensor_map.draft.json`を`tensor_map.json`に昇格する。
3. 同一入力で上流logitsと`laya-candle`のlogitsを比較する。**shapeが合ったことは何の証拠にもならない。**
4. temperatureの扱いをADR化する（スカラー維持か、`temperature_by_options`を持つ型に拡張するか）。
