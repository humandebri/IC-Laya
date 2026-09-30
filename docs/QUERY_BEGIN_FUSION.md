# 開始queryと最初の層の融合

自動分割経路の開始queryで、embeddingと最初のencoder層をまとめて計算する。128トークンでは最初の3層まで進めることで、推論のquery数を11回から10回へ減らした。embedding直後の中間状態を一度クライアントへ返し、次のqueryで読み戻す処理も省ける。

## 実測結果

ローカルIC上で、同じモデルpack・128トークン・3マーカーの入力を使い、圧縮を有効にした自動分割を比較した。MBは10進表記。

| 指標 | F32 | INT8 |
| --- | ---: | ---: |
| query数 | 11 → 10 | 11 → 10 |
| 融合した開始queryの命令数 | 4,196,677,937（約41.97億） | 4,350,567,344（約43.51億） |
| 推論全体の命令数 | 41,817,971,224 → 41,662,071,384 | 42,585,022,903 → 42,565,971,134 |
| 通信ペイロード | 8,386,314 → 8,159,158 bytes | 1,383,150 → 1,337,990 bytes |

この入力では、開始queryがencoder 1〜3、続く8回が各3層、最後のqueryがencoder 28と残りの処理を担当する。命令数の削減幅は小さく、主な効果はqueryの往復を1回省けることにある。

通信量は成功した呼び出しのCandidリクエストと応答の中間状態blobの合計。応答のCandid包絡と最後のlogitsは含めない。命令数は圧縮を含むハンドラー内の計測値で、CDKの引数デコードと応答エンコードは含めない。実行時間はqueryキャッシュを制御していないため、速度改善の根拠には使わない。

## 適用と互換性

`tools/canister_infer.py --query-stepped`の自動分割で利用する。INT8は引き続き`--query-int8`で選ぶ。F32の短い入力を既存の単一queryで処理する経路は維持する。

追加した開始APIは、従来の入力に`max_steps: nat32`を加えた通常のqueryである。

- `begin_token_inference_batch_query`
- `begin_token_inference_int8_batch_query`
- `begin_token_inference_batch_compressed_query`
- `begin_token_inference_int8_batch_compressed_query`

開始時のステップ数は1〜16。owner認証と固定packの検査を行い、推論中にupdate callを呼ばない。既存の開始・継続APIは維持し、固定幅を明示した従来経路はembeddingだけの開始queryを使う。

自動分割では開始queryも命令数制限による失敗時に幅を半分へ減らして再試行する。中間状態はクライアントが保持するため、失敗したqueryによる推論状態の更新はない。ローカル実機では128トークンのINT8開始処理を4層にすると制限に達し、2層での再試行は成功して、従来経路と中間状態が完全一致した。

## 検証と限界

100入力のF32・INT8両経路で、変更前とlogitsが完全一致した。各経路で圧縮あり・なしも一致し、通常の自動分割では失敗したqueryはなかった。100入力中の最大計測値は4,753,940,497命令で、いずれもローカルICで完了した。今回の変更による精度の追加劣化はないが、既存のINT8量子化によるF32との判定差は残る。

開始・継続の状態一致、同一リクエストの再試行、範囲外ステップの拒否、非ownerの拒否、不正な圧縮データの拒否、update側の推論jobが変化しないことも確認した。Pythonの82テスト、Candleを有効にしたcanisterの9テスト、workspaceのlocked検証が通った。

この分割設定は検証した入力範囲の設定であり、すべての入力に対する最小query数を証明したものではない。

## 測定資料

レビュー後、命令数制限で縮めた幅を終盤の統合でも守るよう修正した。通常の幅では上記の10 query設定を維持する。検証ツールは全入力の内容とクライアント・検証実装のSHA-256を保存し、再開時に照合する。旧形式の記録や入力・実装の異なる記録は再利用せず、新しい出力先で測定する。

- 集計: `artifacts/query_begin_fusion/summary.json`
- 100入力の各queryの命令数: `artifacts/query_begin_fusion/validation.json`
- プロトコル検証: `artifacts/query_begin_fusion/protocol.json`
- 制限後の再試行: `artifacts/query_begin_fusion/limit-retry.json`
- CLI動作確認: `artifacts/query_begin_fusion/cli-int8.json`
- 変更前の測定: `artifacts/query_transport/validation.json`

測定したWasmのSHA-256は`84d7dfd1ffcf72467cc6af309a80585ea658387e4f2772c8f2937114c409c5d5`。モデルbundleは`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。

詳細なvalidation・protocol・CLI記録はローカル生成物としてGitから除外する。Gitには上記の最終集計を保存し、再検証には固定corpusと`artifacts/references/int8-final-corpus.json`を使う。
