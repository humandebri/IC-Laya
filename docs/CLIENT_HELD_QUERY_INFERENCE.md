# クライアント保持の分割query推論

2026-09-30に、単一のdecision-engine canisterでクライアント保持の分割queryを実装した。重みはcanisterに置き、中間テンソル・入力情報・進行位置をクライアントが保持する。128-token入力を通常queryだけで最後まで計算でき、測定した入力では直接updateとlogitsが完全一致した。

この文書の計測は入力コピー削減前のWasmを対象とする。その後の変更と検証は[INT8変換コストの計測](INT8_CONVERSION_COST.md)を参照。

中間状態もINT8で保持する別経路は[実験用INT8 query推論](INT8_ACTIVATION_QUERIES.md)に記載している。

## 呼び出し方

通常の[ローカルセットアップ](GETTING_STARTED.md)で今回のWasmをbuild・installまたはupgradeし、モデルをwarmupした後に実行する。

```bash
python3 tools/canister_infer.py \
  --identity ic-laya-int8 \
  --input artifacts/laya-choice-128-input.json \
  --query-stepped --query-steps-per-call 2 \
  --output build/query-128-result.json
```

現在のCLIは、`--query-steps-per-call`を省くと入力長に応じて分割幅を選ぶ。指定すれば1〜16段の固定幅になる。大きい値が命令上限内に収まるとは限らない。`--query-compression`の既定値は`auto`で、中間データを可逆圧縮する。従来の生データによる測定を再現する場合は`--query-compression none`を付ける。最新の[実装と測定](QUERY_TRANSPORT_IMPLEMENTATION.md)を参照。

固定幅の互換経路では開始queryはEmbeddingのみ。自動選択の経路では[開始queryに最初の層を融合](QUERY_BEGIN_FUSION.md)し、継続queryは選んだ数の段を実行する。モデルは32段なので、固定1段では開始を含め33回、固定2段では17回のqueryとなる。最終段では中間状態の代わりにlogitsを返す。自動選択のF32経路は16 tokens以下で既存の直接queryを使い、中間データを返さない。

`--project-root`で専用のローカル検証プロジェクトも選べる。`--initialize`、`--pack`、`--warmup`は従来どおりupdateによる準備操作であり、`--query-stepped`がqueryにするのは推論部分。uploadとwarmupの時間は推論所要時間から除く。

中間状態はCLIプロセスのメモリに保持し、送信時だけ一時ファイルのバイナリCandidを使う。F32値をJSONの小数へ変換して次の入力にする処理はない。プロセス終了後に途中から再開する保存機能は初版にはない。

通信タイムアウトなどは同じ中間状態から最大2回再試行する。複数段の呼び出しが命令数上限に達した場合は、同じ状態から段数を半分（端数切り捨て、最小1）に減らして再試行する。1段でも実行できない場合や入力不正の場合はエラーになる。updateへ自動切り替えはしない。

## query API

```text
begin_token_inference_query(TokenInput) -> Result<QueryInferenceProgress>
continue_token_inference_query(blob, max_steps: nat32) -> Result<QueryInferenceProgress>

QueryInferenceProgress =
  Continue { state: blob, completed: nat32, total: nat32, instructions: nat64 }
  Done     { logits: vec float32, completed: nat32, total: nat32, instructions: nat64 }
```

両APIはowner専用で、固定の測定済みpack `bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`を要求する。モデルがcoldなら拒否する。通常queryをクライアントが独立して呼ぶ構成で、composite queryは使わない。

既存の16-token `infer_tokens_query`はそのまま残る。新しいAPIは別経路であり、この既存ガードを解除していない。既存の分割update用`INFERENCE`にも読み書きしない。

CDKのカスタムデコーダーは、Candidのデコード前にownerを確認し、入力バイト数とデコード仕事量を制限する。開始は4096 bytes、継続は1,100,000 bytes以内のCandid引数を受け付ける。継続状態は長さ・進行位置・bundle・revision・token・マーカー・非有限値を検証してからTensorとして復元する。デコード前のowner拒否や不正Candidはqueryのrejectとなり、モデル内部の検証エラーはResultのErrとなる。

## 中間状態の形式

state blobは次の順序で並ぶ。整数とF32値はlittle-endianで、Tensorは行優先。形状はモデルconfigと進行位置から導出し、クライアントによる任意の形状指定を受け付けない。

| Offset | 値 |
| ---: | --- |
| 0 | magic `LAYQ`、4 bytes |
| 4 | format version、u32、現在1 |
| 8 | inference revision、u32、現在1 |
| 12 | model bundle、32 bytes |
| 44 | 次に実行する段、u32 |
| 48 | qtype ID、u32 |
| 52 | token数、u32 |
| 56 | マーカー数、u32 |
| 60 | token IDs、続いてmarker位置、各u32 |
| その後 | 中間TensorのF32値 |

encoderから最後のdecision層の直前まではtokens × hiddenのTensorを渡す。最後のdecision層の後はmarkers × hiddenになる。scorerを終えた状態は継続状態としてexportできない。

モデルが交換されたらbundle不一致で拒否する。推論の意味や段の区切りを変更するupgradeではinference revisionを増やす必要がある。クライアントCLIは実行前後でWasmとpackが変化していないことも確認する。

stateには認証情報を付けていない。クライアントが値や進行位置を変更できるため、raw logitsを得る用途のAPIであり、既存Receiptやexecutorの実行許可には使わない。

## ローカル測定結果

正確なWasm・pack・入力hashは検証記録（ローカル生成物: `artifacts/client_held_query/validation.json`）に保存した。同じWasm、warm済みpack、同じ128-token入力で各方式を10回測定し、実行順を交互にした。全試行（ローカル生成物: `artifacts/client_held_query/benchmark.json`）には各queryの命令数・引数サイズ・継続blobサイズ・所要時間を含む。

| 方式 | 推論の呼び出し回数 | 所要時間の中央値 |
| --- | ---: | ---: |
| 直接update | 1 | 5.43秒 |
| 分割update、16段ずつ | 2 | 5.82秒 |
| 分割query、2段ずつ | 17 | 4.09秒 |

queryの同一呼び出しキャッシュと同時負荷は制御していない。この測定はCLI起動・直列化・応答の解析を含むローカルの往復時間であり、mainnetの所要時間や純粋な演算速度を表すものではない。別CLIからの2段/query試行（ローカル生成物: `artifacts/client_held_query/choice-128-two-steps.json`）では14.32秒となっており、所要時間のばらつきもある。

2段/queryの128-token推論では、中間blobは通常524,872 bytes。要求Candidと返却された継続blobを合わせて約16.8MBを転送する。返却Candidの包みと最終logitsのサイズはこの合計に含めていない。1段では約32.6MBとなった。命令数は直接updateより少し増え、総命令数の削減は確認されなかった。

2段/queryの最大内部計測値は約2.712B（27.12億）命令だった。10回とも各queryがICの上限内で完了した。build・テストと追加確認の記録（ローカル生成物: `artifacts/client_held_query/checks.json`）には、最終Wasmのhash照合と、upgrade直後のcold状態での拒否も残した。

応答の`instructions`は推論helper内の計測で、stateの復元・exportを含む。owner/pack確認とCDKによる引数デコード・戻り値エンコードは含まない。IC全体の厳密な計測値として扱わず、実queryが上限内で完了した事実と合わせて読む。

## 検証と再現

```bash
cargo test --workspace --locked
cargo test -p decision-engine --features candle --lib --locked
cargo check -p decision-engine --features candle --locked
.venv/bin/python -m unittest discover -s tests -v
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine

# 対象はownerが一致し、今回のWasmとpackをロード済みのローカルcanister。
python3 tools/check_client_held_query_protocol.py \
  --project-root build/client-query-project \
  --identity ic-laya-query-local-test --repeats 10
```

検証ツールはinstall・upload・upgrade・reinstallをしない。queryとupdateの数値を比較し、独立性検証のため既存の分割update jobを開始するので、専用の検証canisterで実行する。テストidentityはネットワーク初回起動前に作成してローカル初期残高を得る。大きいpackのwarmupには追加のローカルcanister top-upが必要な場合がある。

検証済み項目:

- prenorm/postnormのtiny INT8 fixtureで全境界のstateが無損失で往復し、3 qtypes・1段/2段のlogitsが直接推論と一致。
- 実packで16・17・28・64・128 tokens、3 qtypes、2/3/7 markersの代表入力が完走し、logits一致。全組み合わせの精度や命令数を保証する測定ではない。
- 不正なversion、revision、bundle、step、入力長、token、NaN、欠損・余分なbytesを拒否。Infinityもコアのテストで拒否。
- 同じ状態からの再試行、2件の推論の交互進行、既存update jobとの分離、非owner拒否、従来16-token queryの受付と17-tokenガード。
- クライアントの通信失敗時の同一payload再試行と、2段の命令数超過から1段へ減らす処理。

実装時の[計画](CLIENT_HELD_QUERY_INFERENCE_PLAN.md)も残している。複数canisterへの重み分散や中間状態の圧縮は追加していない。
