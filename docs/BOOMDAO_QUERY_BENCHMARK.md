# BOOM DAO提案のqueryベンチマーク再測定

2026-09-30、現在の開始query融合・自動分割・可逆圧縮を使って、ローカルcanisterで測定し直した。過去のSNS実験と同じ`tools/sns_proposal_triage.py`の質問・選択肢・構造化要約を使う。公開Dashboard APIから提案617・620・653を取得し、JSONとtoken入力を保存した。提案本文全体を投入した測定ではない。

## 結果

各入力をF32、INT8の順で交互に3回実行した。命令数と通信量は中央値。命令数はハンドラーのみで、CDKの引数デコードと応答エンコードを含めない。通信量は成功したCandidリクエストと中間状態blobの合計で、HTTP、応答のCandid包絡、最後のlogitsを含めない。

| 提案 | tokens | query回数 F32/INT8 | 総命令数 F32→INT8 | 通信 bytes F32→INT8 | 判定 F32/INT8 |
| --- | ---: | ---: | ---: | ---: | --- |
| 617 | 44 | 5 / 5 | 134.45億 → 151.42億 | 1,278,650 → 208,816 | unlikely / unlikely |
| 620 | 42 | 4 / 4 | 127.69億 → 143.39億 | 911,521 → 149,689 | unlikely / unlikely |
| 653 | 35 | 4 / 4 | 107.41億 → 116.42億 | 757,681 → 126,687 | likely / likely |

INT8の通信量はF32比83.28〜83.67%減ったが、総命令数は8.39〜12.62%増えた。すべて再試行なしで完了し、3回のlogitsは各経路内で完全一致した。最大1 queryの命令数はF32で39.33億、INT8で39.30億だった。

ローカル壁時計時間の中央値はF32で0.150〜0.240秒、INT8で0.123〜0.190秒。queryキャッシュを制御していないため、計算速度の改善や本番環境での遅延の根拠には使わない。

## モデルの判断

617の質問は投票参加の減少・投票権の集中を問うもので、要約は最低投票dissolve delayが1日から20,000日へ増えるという内容。F32・INT8とも`unlikely`を選び、以前のネイティブ測定で見られた問題は解消していない。数値ルールの`critical_review`はモデルと独立した結果である。

620の要約は1日から2日への変更で、両経路が`unlikely`、数値ルールが`standard`を返した。これは提案作成時のrenderingを使用した判定であり、実行直前の状態遷移を再構成した結果ではない。653の要約は250,000,000 tokensを1口座へmintする内容で、両経路が`likely`、ルールが`review`を返した。

現在のソースからネイティブ実行ファイルを再buildして同じ入力も測定した。3件の選択ラベルはcanisterと一致したが、ネイティブとcanister F32のlogits最大絶対差は617で0.158503、620で0.131218、653で0.084889だった。原因は今回の測定では切り分けていないため、両実行環境の数値一致を主張しない。

この3提案は既知の事例を再確認する実験で、代表的な正解ラベル付き精度ベンチマークではない。これらのモデル出力は危険確率や実行認可として使わない。

## 再現と記録

Gitには[最終集計](../artifacts/boomdao_query_benchmark/summary.json)を保存する。全queryの記録と取得snapshotはローカルに残し、コミットしない。集計には元の詳細記録・snapshot・入力・実装のハッシュを含める。

```bash
cargo build -p laya-candle --bin laya-infer --release --locked
.venv/bin/python tools/benchmark_boomdao_queries.py \
  --output-dir artifacts/local/boomdao-rerun --repeats 3
```

既にwarmup済みの専用ローカルcanisterと`checkpoints/laya-int8`を使用する。ツールはモデルのinstall、upgrade、upload、warmupを行わず、推論は通常queryだけで進める。取得する公開提案の内容が変わる可能性があるため、比較には保存されたsnapshotとそのSHA-256を使う。既存の`benchmark.json`がある出力先は上書きしない。

- 全測定結果と各queryの命令数（ローカル生成物: `artifacts/boomdao_query_benchmark/benchmark.json`）
- 617の取得JSON（ローカル生成物: `artifacts/boomdao_query_benchmark/proposal-617.json`）
- 620の取得JSON（ローカル生成物: `artifacts/boomdao_query_benchmark/proposal-620.json`）
- 653の取得JSON（ローカル生成物: `artifacts/boomdao_query_benchmark/proposal-653.json`）
- [測定ツール](../tools/benchmark_boomdao_queries.py)

Wasm: `84d7dfd1ffcf72467cc6af309a80585ea658387e4f2772c8f2937114c409c5d5`。bundle: `bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。入力、tokenizer、取得データ、ネイティブ実行ファイル、Python実装のハッシュも全測定結果に保存した。
