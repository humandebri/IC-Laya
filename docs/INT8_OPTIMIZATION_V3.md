# INT8推論の追加最適化（モデル変更なし）

2026-09-25、固定revisionの実Laya W8A8 packをlocal canisterで再計測した。
weight、tokenizer、入力、量子化方式は変えていない。最終WasmのSHA-256は
`5eacb8c702a71c7e87991e63da361ff4a8f32eed347b09d8d742001669217814`、
packのbundle SHA-256は
`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。

## 変更

- 同じ入力長・head次元・thetaのRoPE用cos/sin Tensorを再利用する。CPUスレッドごとに
  最大4組を保持し、別の長さや設定で無制限に増えないようにした。演算順は変えない。
- 最後のdecision層では、scorerが読むmarker行だけのattention出力とMLPを計算する。
  key/valueは全tokenのままで、最後の層以前も全tokenを計算する。4種類の小型fixtureで
  全行計算後のmarker選択とF32値が完全一致した。
- INT8行列積のtokenタイルを16行から64行に拡大した。32行、64行を順に実測して採用した。
  128行はIC0505（生成Wasm関数の複雑度1,069,970が上限1,000,000を超過）で
  install時に拒否されたため撤回した。128行×8出力も実測したが51.304B命令へ
  悪化したため撤回した。出力行タイルは16行のまま。

## 実モデルの命令数

1Bは10億命令。128-tokenの基準は直前のF32書戻し版、短い3入力の基準は
それ以前に採取したV2版であり、短い入力の削減率にはF32書戻し改善も含む。

| 入力 | 基準 | 今回 | 削減 | 根拠 |
|---|---:|---:|---:|---|
| Choice 38 tokens | 13.338B | 12.459B | 6.59% | JSON（ローカル生成物: `artifacts/int8_optimization_v3/choice-38.json`） |
| Noul 45 tokens | 15.881B | 14.857B | 6.45% | JSON（ローカル生成物: `artifacts/int8_optimization_v3/noul-45.json`） |
| Score 37 tokens | 12.959B | 12.100B | 6.63% | JSON（ローカル生成物: `artifacts/int8_optimization_v3/score-37.json`） |
| Choice 128 tokens | 45.578B | **42.842B** | **6.00%** | JSON（ローカル生成物: `artifacts/int8_optimization_v3/choice-128.json`） |

4入力とも以前のcanisterのserialized logitsと完全一致した。128-tokenは
23.389B＋19.452B命令の**2 update**で完走した。ローカルのwall timeは
5.13秒だったが、mainnetの速度見積もりには使わない。

128-token Choiceで、RoPE再利用とmarker計算を追加した16行タイルは44.261B、
32行タイルは43.404B、採用した64行タイルは42.842Bだった。
16行（ローカル生成物: `artifacts/int8_optimization_v3/tile16-128.json`）・
32行（ローカル生成物: `artifacts/int8_optimization_v3/tile32-128.json`）・
64行（ローカル生成物: `artifacts/int8_optimization_v3/choice-128.json`）。
各候補のpack/input hashとlogitsは同じ。
128行×8出力の不採用結果（ローカル生成物: `artifacts/int8_optimization_v3/tile128x8-128.json`）。

## 呼出し境界とquery

Choiceの質問・選択肢を固定して本文tokenだけを調整した単一updateでは、
113〜118 tokensが成功し、119〜128 tokensは40B命令上限で失敗した。
最大成功実測は**118 tokens、39.950B命令**。残り約0.05Bしかないため、
任意の118-token入力に対する成功保証ではない。
境界測定（ローカル生成物: `artifacts/int8_optimization_v3/update-limits.json`）。

同じChoice schemaではprefixが26 tokensあり、本文1 tokenと終端SEPを加えた
最短の有効入力は28 tokens。この入力でも**9.301B命令**だった。
短い入力の測定（ローカル生成物: `artifacts/int8_optimization_v3/shortest-choice-costs.json`）。
ICPのquery上限5Bには届かない。現APIも推論はupdateのみである。
この結果は当該schema・packでの測定であり、他のschemaの最短値ではない。

## 検証範囲

- 実checkpointのnative版4入力は変更前後でlogits完全一致。canister版も旧版と4/4一致。
- `laya-candle`のfixtureテスト12件、Candle有効の`decision-engine`テスト3件に成功。
- 実Wasmの整数参照760件、量子化144件、F32書戻し216件に成功。
- 128-token継続推論の再送、入力衝突、権限、完了状態保持を検査。
  検証記録（ローカル生成物: `artifacts/int8_optimization_v3/validation.json`）・
  プロトコル結果（ローカル生成物: `artifacts/int8_optimization_v3/protocol.json`）。

測定対象はowner専用raw-token推論。実務データの正答率、`evaluate`とexecutor経路、
mainnetの時間・費用・成功率はこの測定からは確定しない。
