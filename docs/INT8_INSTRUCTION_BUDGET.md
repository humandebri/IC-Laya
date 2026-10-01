# 推論CLIの命令予算による振り分け

`tools/canister_infer.py`に`--max-update-instructions`を追加した。入力token数から現行INT8モデルの命令数を推定し、予算と余裕に収まる場合は既存の`infer_tokens`を1 updateで呼ぶ。超える場合は既存の分割推論を使い、予算に応じて1・2・4・8・16ステップ単位を選ぶ。推定にはモデル推論を追加しない。

```bash
python3 tools/canister_infer.py \
  --input artifacts/laya-choice-128-input.json \
  --max-update-instructions 20000000000 \
  --output artifacts/budget-result.json
```

対象はWasm `0xdd7013df97f0b2b039540cb94888aebd7239efe085d53935601f6206236d1b49`、pack `bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。両hashを実行前に照合し、どちらかが異なれば予算付き実行を拒否する。入力は1〜128 tokens、選択肢マーカー2〜7個、qtype 0〜2に限る。

単一updateの推定式は`1.2B + 0.31B × tokens`。既存96入力と実用例16入力のすべての実測を上回る値として選んだ。指定予算が35Bより大きくても、振り分けには35Bを上限として使う。分割幅の予算閾値は現行128-tokenの1ステップ計測（ローカル生成物: `artifacts/int8_optimization_v4/budget-profile-choice-128.json`）から設定した。各幅の測定最大値と採用閾値は次の通り。

| ステップ/update | 測定最大 | 採用閾値 |
|---:|---:|---:|
| 1 | 1.358B | 2B |
| 2 | 2.716B | 4B |
| 4 | 5.431B | 7B |
| 8 | 10.845B | 14B |
| 16 | 21.665B | 28B |

指定できる最小予算は2B。実行中はcanisterが返した各updateの計測値を予算と比較し、超えた場合は後続ステップを呼ばずエラーにする。これは**実測に基づくソフト予算**であり、実行前に命令数を数学的に保証するものではない。実際の超過が起きたupdateの命令は既に消費されている。上記計測値はcanisterが返す内部計測で、呼出し全体の厳密な命令数ではない。Wasmやpackを変更したら再測定して推定式と閾値を更新する。

予算は1 update当たりの上限で、推論全体の命令数上限ではない。128-tokenの上記入力では、35B予算の分割実行が合計39.267B命令、2B予算の分割実行が合計39.578B命令だった。小さい予算ほどupdate回数と分割処理の費用が増える。

## local canister検証

| 入力・予算 | 経路 | update回数 | 最大1回の計測値 | logits |
|---|---|---:|---:|---|
| 103 tokens、35B（ローカル生成物: `artifacts/int8_optimization_v4/budget-direct-103.json`） | 単一 | 1 | 31.609B | 前回の単一updateと一致 |
| 128 tokens、35B（ローカル生成物: `artifacts/int8_optimization_v4/budget-split-128-35b.json`） | 16ステップずつ | 2 | 21.442B | 下行と一致 |
| 128 tokens、20B（ローカル生成物: `artifacts/int8_optimization_v4/budget-split-128-20b.json`） | 8ステップずつ | 4 | 10.769B | 上行と一致 |
| 128 tokens、2B（ローカル生成物: `artifacts/int8_optimization_v4/budget-split-128-2b.json`） | 1ステップずつ | 33 | 1.358B | 上行と一致 |

予算を指定しない既存CLIの直接・`--stepped`・`--profile`の動作は変更していない。予算指定時は振り分けが分割幅を決めるため、これらの手動指定と同時には使えない。
