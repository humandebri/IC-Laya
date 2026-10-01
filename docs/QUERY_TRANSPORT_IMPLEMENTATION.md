# queryの自動分割と可逆圧縮

この文書の数値は開始処理を独立したqueryにしていた版の測定。その後、開始処理と最初の層を融合し、128 tokensの既定経路を10 queryへ減らした。[開始queryの融合](QUERY_BEGIN_FUSION.md)を参照。

`feat/client-held-query-inference`で、入力長に応じたqueryの分割と、中間状態の可逆圧縮を実装した。推論は独立した通常queryだけで進め、中間状態はクライアントが保持する。圧縮はcanisterでの返却と次の要求の両方に使う。updateへの切り替えや他canisterへの呼び出しはない。upload・warmupと管理情報の取得は別の準備・検証操作である。

## 実モデルの結果

128 tokens・3選択肢について、同じWasm上の生データ・2段固定と比較した。通信集計は成功したCandidリクエストと返信のcontinuation blobの合計で、HTTP/TLS、返信のCandid包み、最終logitsは含まない。

| 指標 | F32 activation | INT8 activation |
|---|---:|---:|
| query回数、2段固定→自動 | 17→11 | 17→11 |
| 通信、生データ2段固定 | 16,796,736 bytes | 4,230,208 bytes |
| 通信、自動分割＋圧縮 | 8,386,314 bytes | 1,383,150 bytes |
| 通信削減率 | 50.07% | 67.30% |
| 総handler命令数、自動分割＋圧縮 | 41,817,971,224 | 42,585,022,903 |
| 圧縮・展開の追加命令数、自動分割の生データ比 | 2,547,706,552（6.49%） | 481,546,178（1.14%） |
| 最大query handler命令数 | 4,311,213,032 | 4,390,780,587 |

軽い圧縮でも命令数は増える。削減するのは通信量と往復回数で、行列積の回数ではない。高帯域・低遅延の環境で圧縮の追加コストを避けたい場合は、`--query-compression none`を選べる。ローカルquery cacheを制御していないため、壁時計時間から本番の速度は断定しない。

| 入力 | INT8のquery回数 | INT8の通信集計 |
|---|---:|---:|
| 43 tokens、3選択肢 | 5 | 207,103 bytes |
| 65 tokens、3選択肢 | 6 | 360,710 bytes |
| 128 tokens、3選択肢 | 11 | 1,383,150 bytes |
| 128 tokens、7選択肢 | 11 | 1,333,883 bytes |
| 16 tokens、7選択肢 | 3 | 25,523 bytes |

F32 activationの自動選択では、16 tokens以下を既存の`infer_tokens_query`へ送る。16 tokens・7選択肢は1 query、Candid要求127 bytes、handler命令数4,726,291,272で完了し、中間状態の通信はなかった。最終logitsと返信の包みは別に送られる。INT8 activationを指定した場合は短くてもその実験経路を維持し、F32へ自動切り替えない。

## 分割の選び方

CLIで`--query-steps-per-call`を省略すると、測定した固定packの入力長とactivation形式からencoderの初期幅を選ぶ。

| 入力長 | F32 | INT8 |
|---|---:|---:|
| 1〜16 | 直接query | 16段 |
| 17〜43 | 9段 | 8段 |
| 44〜65 | 6段 | 6段 |
| 66〜128 | 3段 | 3段 |

最後のencoderと軽い終盤処理を、安全側の条件でまとめる。128 tokensでは3段ずつ9回、その後の5段を1回で処理し、Embeddingの開始queryを含めて11回になる。これは任意の入力に対する最小query数の保証ではなく、測定範囲から選ぶ保守的な条件である。入力の切り捨てや選択肢の削減は行わない。

実際に命令上限へ達した場合は、同じ中間状態から段数を半分にして再試行し、その推論の後続幅にも上限を反映する。通信失敗は同じバイト列を再送する。1段でも実行できなければエラーを返し、updateへ切り替えない。失敗した要求のCandidバイト数は`failed_attempt_request_candid_bytes`へ別に記録し、成功分の通信集計と混ぜない。送信失敗が実際にどこまでネットワークへ届いたかは、この値だけでは分からない。

100入力の自動選択はすべて命令上限内で完了し、再試行はなかった。通常の分割queryの最大handler計測値は4,753,778,888だった。63/127 tokensなどではINT8の端数処理で128 tokensより重いqueryがある。handler counterはCDKの引数デコードと返信エンコードを含まない。測定済み入力の成功は、全入力に対する余裕の保証ではない。より小さい幅を使う場合は`--query-steps-per-call 2`などで指定できる。

自動幅は固定packと32段に結び付ける。クライアントは開始応答のbundleと段数を確認し、未知のモデルへこの条件を適用しない。Wasmやkernelを変更した場合は再計測する。

## 可逆圧縮の形式

圧縮用のbegin/continue queryを追加した。引数と戻り値の型は既存の分割queryと同じ。元の生データ用APIも維持する。

```text
begin_token_inference_compressed_query(TokenInput)
continue_token_inference_compressed_query(blob, max_steps: nat32)
begin_token_inference_int8_compressed_query(TokenInput)
continue_token_inference_int8_compressed_query(blob, max_steps: nat32)
```

stateの圧縮包みはlittle-endianの16-byte headerとzlib streamからなる。

| Offset | 内容 |
|---:|---|
| 0 | magic `LAYZ` |
| 4 | transport version、u32、現在1 |
| 8 | codec、u32、1はINT8＋zlib、2はF32 byte shuffle＋zlib |
| 12 | 展開後のバイト数、u32 |
| 16 | zlib stream |

F32では、入力情報までのheaderを保ち、各F32値の4 bytesをbyte位置ごとに並べ替えてから圧縮する。復元時には元の並びへ戻す。FP16への変換や再量子化は行わない。INT8ではscaleを含む元のバイト列をそのまま圧縮する。Wasmは`flate2 1.1.10`のRust backend、`miniz_oxide 0.9.1`、`Compression::fast()`を使う。候補比較で使ったPython gzipと実装のzlib streamは異なるため、実装後の表を通信量の根拠とする。

圧縮後の包みが元より大きい場合は生の`LAYQ`/`LAYI`を返す。圧縮用のcontinue queryは生データも受け付ける。クライアントは応答を展開してheaderを検証するが、次の送信には受け取った圧縮バイト列をそのまま使う。再圧縮やF32値の小数文字列化は行わない。

送信前・展開後とも状態サイズを既存の上限1,049,176 bytes以内に制限する。RustとPythonの展開処理は、申告した長さ＋1 bytesまでしか出力を読み取らない。申告長との不一致、未知のversion/codec、不正なzlib checksum、truncated stream、末尾の余分なデータを拒否する。展開した状態には、さらに既存のbundle・revision・shape・token・scale・非有限値の検証を適用する。各APIは従来どおり固定pack・owner専用で、Candidをデコードする前にcallerと要求サイズを確認する。

## 使い方

専用のlocal canisterへCandle有効のWasmをupgradeし、既に保存しているモデルをwarmupした後に使う。モデルの再uploadは不要。

```bash
python3 tools/canister_infer.py \
  --project-root build/client-query-project \
  --identity ic-laya-query-local-test \
  --input artifacts/laya-choice-128-input.json \
  --query-stepped --query-int8 \
  --output build/query-compressed.json
```

`--query-int8`を省くとF32 activationのまま自動分割・可逆圧縮を使う。`--query-compression none`は圧縮を無効にし、自動分割は残す。`--query-steps-per-call 2 --query-compression none`は従来の2段固定・生データの条件になる。

低水準の`QuerySession`は既存呼び出しとの互換のため、引数なしでは1段・生データのまま。自動選択と圧縮は`QuerySession(icp, inp, steps=None, activation_format="int8", compression="auto")`で選ぶ。通常CLIと`infer_queries`の既定は自動選択・圧縮である。

## 検証

実Wasm上で、既存の96入力に追加4入力を加えた計100入力をF32・INT8の両経路で比較した。圧縮ありと生データのlogitsは全件完全一致し、既存96入力ではモデル・corpusに結び付いた過去のF32/INT8参照とも完全一致した。既存のINT8とF32の判定差8/96件は残る。この圧縮・分割変更が判定差を追加したわけではなく、量子化の差を改善したわけでもない。正解ラベルによる精度評価は含まない。

実canisterで、圧縮状態の境界のバイト列が生データ経路と完全一致すること、同一要求の再試行、非ownerの拒否、破損・過大な展開・展開爆弾の拒否、queryが既存のupdate jobを変更しないことを確認した。通常CLIの既定オプションでも128-tokenのINT8推論を完走した。Python80 tests、Candle有効のdecision-engineの8 tests、既定featureのworkspaceテストを通過した。

最終Wasmは`0x6a73e1ebc2d9b3084b68c64cab9c43c995361701b5e991aa99009410b756df2a`、packは`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。全100入力の測定（ローカル生成物: `artifacts/query_transport/validation.json`）、protocol検証（ローカル生成物: `artifacts/query_transport/protocol.json`）、CLIの確認（ローカル生成物: `artifacts/query_transport/cli-int8.json`）、[集計](../artifacts/query_transport/summary.json)を保存した。

```bash
python3 tools/validate_query_transport.py --corpus --output build/query-transport-new.json
python3 tools/check_query_transport_protocol.py --output build/query-transport-protocol-new.json
```

これらのツールは、モデルがロード済みの専用local canisterを使う。準備が必要な場合だけ検証ツールに`--warmup`を付ける。過去の測定結果を上書きせず、新しい出力先を指定する。
