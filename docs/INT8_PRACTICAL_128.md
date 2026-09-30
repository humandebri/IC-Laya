# Layaの短い実用分類をlocal canisterで試す

2026-09-25に、固定した英語の例16件を現行INT8 pack・Wasmのlocal canisterへ送り、事前に付けたラベルと比較した。再実行にはPython環境に`tokenizers`を入れ、`python tools/check_practical_laya.py`を使う。スクリプトは既存canisterのowner専用`infer_tokens`を1回のupdateで呼び、モデルのupload・installは行わない。

| 用途 | 一致 | 観察 |
|---|---:|---|
| サポート窓口の振り分け | 8/8 | 請求・技術サポート・解約を選択。長文2件も一致 |
| 解約依頼の有無 | 3/4 | 「次回更新を止める」を`no`と判定 |
| セキュリティ報告の緊急度 | 3/4 | 未知のログインを`medium`ではなく`low`と判定 |
| **合計** | **14/16** | 代表性のない小規模診断 |

全件29〜103 tokensで128-token上限内。推論はすべて**1 update**で完走し、8,945,450,812〜31,609,139,020命令だった。raw logits、選択肢、入力ID、命令数、Wasm・pack hashは単一updateの結果JSON（ローカル生成物: `artifacts/int8_optimization_v4/practical-128-probe.json`）に保存した。先行する2 update測定（ローカル生成物: `artifacts/int8_optimization_v4/practical-128-stepped-probe.json`）と入力、Wasm・pack hash、全16件のlogits・判定が一致した。単一updateは分割実行より1件当たり6,002,700〜12,959,129命令少ない。

誤判定の2件はどちらも僅差だった。解約依頼の`no`対`yes`は0.0666、緊急度の`low`対`medium`は0.0536のlogit差。これらの差は正答確率ではない。特にセキュリティ報告の自動処理には使えない。

これは手作りの小さな英語例を使った探索で、無作為抽出した実案件や独立したholdoutではない。日本語、表現の揺れ、曖昧な依頼、攻撃的な入力、運用中の誤判定率は測っていない。`infer_tokens`はowner専用のraw logitsを返し、認証付きReceiptは発行しない。128 tokensちょうどの自然文は今回の品質診断には含めていない。任意の128-token入力が1 updateに収まる保証はなく、超過する入力には既存の分割推論を使う。既存の128-token合成入力の実行境界は[最適化記録](INT8_OPTIMIZATION_V4.md)を参照。
