# クライアントに中間状態を保持する分割query推論計画

作成日: 2026-09-30。状態: API・クライアントを実装し、ローカル実packで検証済み。
作業ブランチ: `feat/client-held-query-inference`。

実装結果と再現方法は[分割query推論レポート](CLIENT_HELD_QUERY_INFERENCE.md)、測定値は検証記録（ローカル生成物: `artifacts/client_held_query/validation.json`）を参照。以下は実装前の計画を保存したもの。初版では継続状態を構造化されたCandid recordではなく、version付きのバイナリblobで渡す。1段・2段のqueryを検証し、クライアントのメモリに状態を保持する。

## 目的と対象

128 tokensまでの入力について、推論を複数の独立した通常queryで完了する。モデルの重みは単一のdecision-engine canisterに置き、クライアントが中間テンソルと進行位置を保持する。入力文を分割するのではなく、入力全体に対する層の実行を分割する。

最初の対象は測定済みのW8A8 packとowner専用のraw logits APIとする。モデルのupload・warmupは従来のupdateを使う。既存の直接update、canister内に状態を置く分割update、16-token queryは比較対象として維持する。複数canisterへの配置、composite query、認証付きReceiptの生成、executorとの連携、128 tokens超への拡張は今回の実装範囲に含めない。

成功条件は、長い入力をqueryだけで最後まで計算でき、元の推論と同じlogitsを得られること。高速化は実測によって判断し、総命令数の削減は前提にしない。

## 既存実装と実測から分かること

- `crates/laya-candle/src/lib.rs`の`InferenceSession`はbundle、TokenInput、hidden Tensor、nextを保持する。`begin_inference`でEmbeddingを計算し、`step_inference`で1段ずつ進められる。
- 測定対象モデルは28 encoder層、final normとqtype加算、2 decision層、scorerの計32段。初期Embeddingはこの32段の外にある。
- 最後のdecision層はマーカー行だけを出力する。途中状態の形状は常にtokens × hiddenではなく、終盤でmarkers × hiddenへ変わる。
- `canisters/decision-engine/src/lib.rs`の`TokenJob`はupdate用の状態をcanisterに保存している。新しいquery経路はこれを使用しない。
- `tools/inference_budget.py`は既存の128-tokenプロファイルについて、連続1段の最大を約1.358B、2段を約2.716B、4段を約5.431B命令と記録している。新しいqueryの直列化コストを含む測定値ではない。
- 既存128-token分割実測（ローカル生成物: `artifacts/int8_optimization_v4/budget-split-128-2b.json`）と[短入力query実測](INT8_SHORT_QUERY.md)を基準にする。現在のraw queryは16 tokensまでで、128 tokensは推論前に拒否される。

既存値からは、最初に1段/queryで成立を確かめ、2段/queryを次に測る方針が妥当である。4段固定は既存の推論本体だけで5Bを超えるため採用しない。軽い段と重い段を組み合わせた可変分割は初期検証後に判断する。

## APIと中間状態

追加APIの仮名は次のとおり。どちらも通常queryであり、クライアントがそれぞれ独立して呼び出す。

| API | 入力 | 出力 |
| --- | --- | --- |
| `begin_token_inference_query` | TokenInput | Embedding計算済みの継続状態、計測情報 |
| `continue_token_inference_query` | 継続状態、希望する段数 | 次の継続状態、または最終logits、計測情報 |

戻り値は`Continue`と`Done`のvariantにし、完了時にはlogitsだけを返す。開始queryはまずEmbeddingのみとする。継続queryの許可段数は初期実装では1とし、実queryで予算を検証した後に2を追加する。クライアントの希望段数だけで実行範囲を決めず、canister側の許可範囲でも検証する。

継続状態に含める情報:

- `format_version`と推論実装の互換性を表す`inference_revision`。
- `bundle`。受信時にロード済みモデルと一致することを確認する。
- 検証済みTokenInput。最大128 tokensなので、初版はtoken IDsを含めて引き継ぎ、既存検証を再利用する。
- `next_step`。クライアントの「層番号」はencoderだけでなくfinal norm、decision、scorerを含む進行位置として定義する。
- `hidden`。行優先のF32 little-endianバイト列をCandidのblobで受け渡す。INT8なのは重み・主要演算の方式であり、継続状態を独自にINT8化して精度を変えない。

テンソルの列数はモデルconfig、行数はTokenInputとnext_stepから導出し、任意の形状指定を受け付けない。最終decision層の前後を含めて正確な形状規則を定義する。F32の生バイトを使い、JSONの小数文字列への変換を中間経路に挟まない。

各応答には完了段数、総段数、当該query内で測れた命令数、実装revisionを付ける。累積命令数・総往復数・総転送量はクライアントが集計する。canister内にjob IDやリプレイ用キャッシュは作らない。

## 検証と予算の境界

受信した状態について、owner、モデルのwarm状態、bundleとrevision、入力長・token IDs・qtype・マーカー、next_step、段数、blob長、非有限値を検証する。サイズ計算にはchecked arithmeticを使い、不正な長さのblobから巨大なTensorを確保しない。Candidデコード段階のメモリ消費にも上限を設けるか、利用中のCDKの既存上限を確認する。

canisterから返した中間状態でも、次のリクエストでは未検証の入力として扱う。bundleや入力のハッシュは取り違えの検出に使えるが、クライアントが正しく全段を実行した証明にはならない。このAPIはraw logitsの取得用とし、返された結果を既存Receiptや実行許可として扱わない。

queryは継続状態を永続化しない。同じ状態・段数・互換実装での再試行は同じ結果になることを確認する。モデルが交換されたら拒否し、クライアントは入力から再開する。upgrade後も互換性を推測せず、実装revisionを確認する。warmupが必要な場合は明示的にエラーを返す。

1回のqueryには、入力の復号・検証、Tensor復元、層の実行、Tensorの取り出し、応答の符号化が含まれる。層の命令数だけを5Bに収めても成立するとは限らない。初期目標は余裕を持った4B以下とし、最終Wasmで実queryの成功と全体の予算を確認する。

ハンドラー内のinstruction counterだけでは、その前の引数デコードや、戻った後のCDKによる応答エンコードを測り切れない。計測値の範囲を記録し、利用可能ならローカル実行環境の全体計測を併記する。取得できない分を0として扱わない。実queryでの上限内完了を別途検証する。

## 実装順序

1. **コアの継続状態を外部化する。** `crates/laya-candle/src/lib.rs`に、InferenceSessionの安全なexport/importと形状検証を追加する。必要なら専用moduleへ分離する。コアはCandidに依存させず、canister側でwire型へ変換する。
2. **1段/queryのAPIを追加する。** `canisters/decision-engine/src/lib.rs`に開始・継続APIを追加し、`INFERENCE`を読み書きせずに既存のbegin/stepを利用する。最初は固定pack、最大128 tokens、owner限定で検証する。既存queryの16-tokenガードは解除しない。
3. **クライアントの往復処理を追加する。** `tools/canister_infer.py`の既存CLI・呼び出しadapterを確認し、明示的なquery分割モードを追加する。継続状態はクライアントのメモリに保持する。独立した2件を交互に進められることも確認する。
4. **バイナリ転送経路を整える。** 既存の文字列ベースのCLI呼び出しがblob転送に使えるか確認する。大きな状態をシェル引数に展開せず、stdin・ファイル入力またはバイナリ対応agentを用いる。クライアントadapterの選定はこの確認結果で決める。
5. **実queryで予算を測る。** 1段で成立した後に2段を試す。未測定packやWasmへ既存のupdate用係数をそのまま使わない。新しいWasm・pack・revisionを測定結果にひも付ける。
6. **比較結果と利用方法を残す。** 実測JSONを`artifacts/client_held_query/`へ保存し、対応するレポートとCLIの使い方をdocsに追加する。今回の計画ファイルは実施状況と判断結果を更新する。

送信失敗やタイムアウト時は、クライアントが直前の状態を保持して同じqueryを有限回再試行できるようにする。入力不正・bundle不一致は再試行しない。2段の予算超過では同じ状態から1段へ縮小できるようにし、1段でも超える場合は処理を止め、入力・段・Wasmを記録する。明示されていないupdateへの切り替えは行わない。

## 検証計画と完了条件

| 項目 | 検証内容 |
| --- | --- |
| 数値の維持 | tiny INT8のprenorm/postnorm fixtureで全境界をexport/importし、直接推論と同じlogitsになること。最終decision層の行数変化、1段・2段の区切りを含む |
| 実pack | 16/17/28/64/128 tokens、2〜7 markers、3 qtypesの境界・代表入力を実queryで検証する。fixtureの成功と実モデルの成功を区別する |
| 入力検証 | 不正revision・bundle・step・段数、短い/長いblob、NaN/Inf、不正token・マーカー、非owner、cold modelを拒否する |
| 状態の独立性 | 2件の推論を交互に進めても干渉しない。同一queryの再試行でlogitsが変わらず、既存update jobも変化しない |
| 互換性 | 同じ新Wasmの直接update・既存分割updateとのlogits一致、既存16-token queryのガード維持、モデル交換時の継続拒否を確認する |
| 予算 | 実queryが5B以内で完了し、計測範囲を明記する。リクエスト/レスポンスの実バイト数も測り、ICのサイズ制限内に収まること |
| 待ち時間 | 同じ入力とwarm済みモデルで、直接update・分割update・分割queryのend-to-end時間を比較する |

中間F32バイトの往復は無損失なので、同じWasm・packでのlogits完全一致を初期の合格条件にする。差が出たら許容誤差を広げる前に、形状・phase・演算順序・直列化を調査する。完了結果だけでなく必要な境界の内部値を比較できるようにする。

測定ではupload/warmupを推論時間から分け、CLI起動・ネットワーク往復・直列化を含む利用者視点の時間と、推論内部の計測を区別する。方式ごとに少なくとも10回の成功試行を取り、中央値・最小/最大・失敗数を残す。試行順を交互にし、同一queryのキャッシュが測定を歪めていないか確認する。キャッシュを制御できない場合はその条件を記録し、計算が速くなったと解釈しない。

ローカル実測からmainnetの速度は断定しない。mainnet測定を行う場合は別途対象と費用を定める。初期実装の完了条件は、128-tokenの代表入力が分割queryで完走し、logitsが一致し、命令数・転送量・待ち時間を再現可能な形で記録できること。速度が改善しなければ、その結果を残し、既存update経路を引き続き利用できるようにする。
