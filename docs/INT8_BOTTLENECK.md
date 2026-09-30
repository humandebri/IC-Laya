# 実Laya INT8推論のボトルネック計測

2026-09-25、local IC の `decision-engine` で固定した実モデルを計測した。
canister ID は `4caro-hl777-77775-aaaba-cai`、Wasm SHA-256 は
`5eacb8c702a71c7e87991e63da361ff4a8f32eed347b09d8d742001669217814`、
pack bundle SHA-256 は
`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。
以下の B は10億 **Wasm命令**。wall time や mainnet の料金ではない。

## 測定方法と再現性

- 実モデルのChoice入力38・128 tokensを `profile_token_step` で全32ステップ測定。
  各演算は `ic_cdk::api::instruction_counter` で囲んだ。span は重複計上していない。
- 同一128-token入力の通常実行は42.842B、詳細計測は43.152Bと43.160B。
  詳細計測の上乗せは通常実行比で約0.72%、再測定差は0.019%。logitsは一致。
  38-token入力は通常12.459B、詳細12.601B（上乗せ1.14%）。
- 選択肢とprefixを固定して本文だけ伸ばしたChoiceの単一updateも測定。
  入力長以外の文字列内容は同じではないため、長さに対する傾向の確認に使う。

根拠: 38-tokenの詳細記録（ローカル生成物: `artifacts/int8_optimization_v3/profile-choice-38.json`）、
128-tokenの詳細記録（ローカル生成物: `artifacts/int8_optimization_v3/profile-choice-128.json`）、
128-tokenの再測定（ローカル生成物: `artifacts/int8_optimization_v3/profile-choice-128-repeat.json`）、
通常実行（ローカル生成物: `artifacts/int8_optimization_v3/choice-128.json`）。

## 実モデルで費やした命令

比率の分母は **詳細計測の総命令数**。shape は `(token行, 出力行, 入力列)`。
`int8.matmul` には整数積とF32書戻しが含まれる。

| 区分 | 38 tokens | 128 tokens |
|---|---:|---:|
| 総命令数 | 12.601B | 43.152B |
| encoder 28層の全ステップ | 12.005B (95.3%) | 41.254B (95.6%) |
| INT8行列積 `int8.matmul` | 10.577B (83.9%) | **33.353B (77.3%)** |
| 主要3形状のINT8行列積 | 9.393B (74.5%) | **29.700B (68.8%)** |
| attention QK + softmax + AV | 0.662B (5.3%) | 5.261B (12.2%) |
| activation量子化 | 0.231B (1.8%) | 0.769B (1.8%) |
| RoPE | 0.209B (1.7%) | 0.738B (1.7%) |

128-tokenの主要形状別 `int8.matmul` は次の通り。最初の3形状だけで全体の68.8%。

| 形状 | 呼出回数 | 命令数 | 全体比 |
|---|---:|---:|---:|
| `(128, 5248, 1024)` | 28 | 14.120B | 32.7% |
| `(128, 3072, 1024)` | 30 | 8.882B | 20.6% |
| `(128, 1024, 2624)` | 28 | 6.698B | 15.5% |
| `(128, 1024, 1024)` | 29 | 2.853B | 6.6% |

encoderの各層は128-tokenで概ね1.46〜1.48B命令。最後の全行decision層は
1.460B、marker行だけの最終decision層は0.395B、scorerは0.007Bだった。
したがって、最終decision層やscorerだけをさらに速くしても改善幅は小さい。

## 入力長による変化

| Choice入力長 | 通常実行の命令数 | 呼出形態 |
|---:|---:|---|
| 28 tokens | 9.301B | 単一update |
| 38 tokens | 12.459B | 単一update |
| 64 tokens | 20.389B | 単一update |
| 96 tokens | 31.477B | 単一update |
| 112 tokens | 37.340B | 単一update |
| 128 tokens | 42.842B | 2 update、16ステップずつ |

根拠: 最短入力（ローカル生成物: `artifacts/int8_optimization_v3/shortest-choice-costs.json`）、
38-token（ローカル生成物: `artifacts/int8_optimization_v3/choice-38.json`）、
64/96/112-token（ローカル生成物: `artifacts/int8_optimization_v3/length-sweep.json`）、
128-token（ローカル生成物: `artifacts/int8_optimization_v3/choice-128.json`）。
128-tokenだけ2 updateであり、表の差分を厳密な1-token当たり費用とは解釈しない。
38→128 tokensでattention QK・softmax・AVの比率が5.3→12.2%に増える。

## 合成カーネルで確認できる範囲

現行64行タイルの128-token合成ベンチでは、主な3形状のforward全体が
それぞれ0.301B、0.510B、0.254B命令だった。
カーネル記録（ローカル生成物: `artifacts/int8_optimization_v3/current-components.json`）。
同ファイルの `integer_dots` / `f32_writeback` 診断は **16行タイル・scalar書戻し**
で計測する別実装であり、64行タイル・SIMD書戻しの正確な内訳ではない。
その診断では整数積が92〜97%を占めるが、現行カーネルの比率としては使わない。

## 改善を試す順序

1. 実モデルの28 encoder層で反復するINT8行列積、とくに上位3形状を対象にする。
   全INT8行列積を10%削減できれば、他が同じ場合の総削減は128-tokenで約7.7%。
2. 長い入力ではattention QK・softmax・AVを調べる。128-tokenでは計12.2%だが、
   38-tokenでは5.3%であり、短い入力の優先度は低い。
3. activation量子化とRoPEは各2%未満。ここだけを改善してもqueryには届かない。

最短Choice入力28 tokensの9.301Bをquery上限5Bに収めるには、総命令数を
少なくとも46.2%削減する必要がある。この測定だけでは実現できると判断できない。
正答率、`evaluate`/executor経路、mainnet費用は今回の計測対象外。
