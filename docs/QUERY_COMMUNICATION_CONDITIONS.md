# queryの通信量と分割条件の比較

この文書は圧縮機能の実装前の候補比較を保存したもの。その後、canisterでの可逆圧縮とCLIでの自動分割を実装した。[実装とWasm計測](QUERY_TRANSPORT_IMPLEMENTATION.md)を参照。

2026-09-30、`feat/client-held-query-inference`で、固定の実モデルを使って分割幅と可逆圧縮を比較した。128 tokensでは2段固定の17 queryから、可変幅の11 queryへ減らせた。INT8の通信集計は4.23MBから2.64MBへ減り、同じINT8経路のlogitsは完全一致した。gzip level 6を使った中間状態のサイズは約1.40MBだった。圧縮はPython上での候補評価であり、canisterの通信機能としては未実装。

元のF32経路を維持する場合も、128 tokensの通信を16.80MBから10.50MBへ減らせた。F32のバイトを並べ替えてからgzipで可逆圧縮すると、中間状態は約8.18MBになった。どちらの形式も圧縮の往復で元のバイト列と完全一致する。INT8経路に既存の判定差があることと、今回の分割・可逆圧縮が新しい誤差を加えないことは別の評価である。

## 分割を入力に合わせる

各入力・形式を1段ずつ実行して、32段のhandler命令数と境界ごとの状態サイズを記録した。連続する段の計測値の合計が45億命令以下となる分割から、中間状態の送受信バイト数が最小の分割を動的計画法で選び、実際のqueryで完走とlogits一致を確認した。同じ通信量の候補ではquery回数が少ない方を選ぶ。

1段ずつの計測値には境界の変換処理が繰り返し含まれるため、その合計はまとめた処理の保守的な見積もりになる。CDKの引数デコードと返信エンコードはhandler counterに含まれないので、[公式のquery上限50億命令](https://docs.internetcomputer.org/references/resource-limits/)に対して5億命令の余裕を設けた。これは測定した入力の候補選択であり、任意の入力に対する上限保証や、実コストに対する全候補の最適性の証明ではない。

以下は3選択肢の代表入力。MBは10進数。通信は成功したCandidリクエストと返信のcontinuation blobの合計で、返信Candidの包み、最終logits、HTTP/TLSなどを含まない。

| 入力 | 形式 | query回数、2段固定→可変 | 可変分割後の通信 | 最大handler命令数 |
|---|---|---:|---:|---:|
| 43 tokens | F32 | 17→5 | 1.411MB | 44.29億 |
| 43 tokens | INT8 | 17→5 | 0.356MB | 41.98億 |
| 65 tokens | F32 | 17→6 | 2.666MB | 40.94億 |
| 65 tokens | INT8 | 17→6 | 0.672MB | 44.66億 |
| 128 tokens | F32 | 17→11 | 10.498MB | 43.63億 |
| 128 tokens | INT8 | 17→11 | 2.644MB | 43.42億 |

実測で通った分割幅は次のとおり。開始queryは別に1回あり、最後の幅にはencoder以外の軽い処理も含む。

| 入力・形式 | 継続queryの幅 |
|---|---|
| 43 tokens、F32 | 1, 10, 9, 12 |
| 43 tokens、INT8 | 5, 8, 8, 11 |
| 65 tokens、両形式 | 5, 6, 6, 6, 9 |
| 128 tokens、F32 | 2, 3, 3, 3, 3, 3, 3, 3, 3, 6 |
| 128 tokens、INT8 | 3, 3, 3, 3, 3, 3, 3, 3, 3, 5 |

112-tokenのscore入力、45-tokenのnoul入力、最大7選択肢の16/128-token入力も比較した。全7入力・2形式で、1段ずつ・2段固定・可変分割のlogitsが完全一致した。最大7選択肢の128-token入力も11 queryで完走し、最大handler命令数はF32が44.00億、INT8が43.35億だった。

128-token INT8でencoderを4段まとめる要求は、実際に命令上限に達した。同じ中間状態から2段へ減らす再試行で完走し、logitsが一致した。失敗した要求の通信・計算も発生するため、毎回大きい幅から失敗して探す方式は避ける。最初の検証で計測した条件をpack・形式・入力形状に結び付けて再利用し、未検証条件には小さい幅を選ぶ。

今回、owner専用の継続queryの指定範囲を1〜16へ広げ、クライアントの命令上限時の再試行を段数の半減へ一般化した。CLIの既定値は1のまま。大きな幅を指定できることは、上限内で実行できることを意味しない。可変分割の探索は検証ツールに実装しており、通常CLIで未検証入力の分割を自動決定する機能は追加していない。

## 短い入力は直接queryを選ぶ

16 tokens・7選択肢では、既存のF32 activationの`infer_tokens_query`が1回で完走した。handler命令数は4,726,103,380、入力のCandidは127 bytes、中間状態の返却はなく、分割F32のlogitsと完全一致した。最終logitsと返信の包みの通信は別にある。

この入力を分割すると、F32/INT8とも16段＋16段、開始を含め3 queryとなり、通信はそれぞれ約263KB/67KBだった。16 tokens以下では既存の直接queryを優先する価値が大きい。ただし、今回の7選択肢入力の命令上限までの余裕は約2.74億で、可変分割用に設定した45億のhandler予算は超える。厳しい余裕を必要とする運用なら、この短い入力も分割する判断がある。既存APIは17 tokens以上を計算前に拒否する。

## 可逆圧縮の候補

128-tokenの可変分割で受け渡す全境界の状態を対象にした。同じ状態は返却と次回送信に使うため、表では圧縮サイズを2倍している。要求の小さな包みや入力、将来のcodec識別情報は含まない。

| 形式・前処理 | gzip level 1 | level 6 | level 9 |
|---|---:|---:|---:|
| F32、そのまま | 9.009MB | 8.980MB | 8.980MB |
| F32、byte shuffle | 8.359MB | 8.179MB | 8.155MB |
| INT8、そのまま | 1.468MB | 1.405MB | 1.402MB |

F32のbyte shuffleは、各値の4 bytesを同じbyte位置ごとに並べる処理で、F32を別の数値へ変換しない。高位byteなどの共通性を圧縮器が使いやすくなる。復元後の値も元のbit列のまま。

INT8ではlevel 1から6で約64KB減り、6から9では約3KBしか減らなかった。ホスト側の1回分の状態の圧縮時間の合計は約9/30/35msだった。F32 shuffleではlevel 1/6/9が約58/154/340ms。これらはローカルPythonの単回計測であり、Wasmでの命令数や本番の速度を示さない。速度重視の最初の候補はlevel 1、通信優先の候補はlevel 6。level 9の優先度は低い。

canister上で採用する前に、Wasmで圧縮・展開の命令数を測る必要がある。現在の3段queryへ圧縮処理を追加しても上限内に収まるか、返信と次回要求の両方で効果があるかを確認する。圧縮blobの最大長に加え、展開後の最大長も既存の状態上限に制限し、codec/versionを識別する。生の状態形式と互換経路は維持する。一般的なHTTP圧縮を設定するだけで、現在のquery blobの両方向が小さくなるとは仮定しない。

## 通信環境による選択

待ち時間は、計算時間＋query回数×往復遅延＋送受信時間＋codec処理時間で考える。128-token INT8の17→11 queryは、約1.586MBの通信と6回の往復を削る。往復遅延100msなら、往復部分だけで約0.6秒減る。上下方向とも10Mbpsと仮定すると、バイト削減分の送受信時間は約1.27秒。これは回線を一定とした計算であり、本番測定ではない。

| 条件 | 優先する候補 | 残る確認事項 |
|---|---|---|
| 既存F32の判断を維持する | 可変分割、F32 byte shuffle＋軽い可逆圧縮 | canisterのcodec命令数 |
| 現在の実験INT8の判定差を許容できる | 可変分割、INT8＋軽い可逆圧縮 | 正解ラベルによる用途別評価、codec命令数 |
| 低帯域・大きい往復遅延 | query回数を減らし、level 6を比較 | 圧縮追加で分割数が増えないか |
| 高帯域・低遅延、命令数重視 | 可変分割を先に使い、圧縮の採否は実測 | 削減した転送時間がcodec時間を上回るか |
| 16 tokens以下、既存APIの余裕でよい | F32の直接query | 形状・packと上限までの余裕 |

可変分割による総handler命令数の削減は、128 tokensでF32約0.31%、INT8約0.058%だった。主な効果は通信と往復回数で、モデルの行列積が減るわけではない。ローカルquery cacheを制御していないため、反復queryの壁時計時間から本番の高速化は断定しない。

## 今回採用しない案

- INT4やさらに粗いactivation量子化は、既存のINT8で実canister F32と8/96件の判定差があるため優先しない。可逆圧縮では、この既存差を改善も悪化もさせない。判定差は正解率の低下量とは同じではない。
- FP16を通信だけに使う案は、F32状態のおおむね半分まで減らせる候補だが、bit単位の一致を失う。今回の結果には含めず、用途別の精度検証を別に要する。
- 入力から不要な重複を取り除けば状態量と計算を減らせるが、必要な情報を切り捨てたり、入力長だけを変えた別の問題の結果を同じ精度として比較したりしない。
- updateによる既存のcanister保持ジョブなら、中間状態をクライアントと往復せずに済む。通信を入力・ジョブID・結果へ絞る候補だが、queryのみという要件を変更し、合意の待ち時間とcyclesを比較する必要がある。
- canisterの分割だけでは、中間状態の総データ量は消えない。単一のcomposite queryで全計算を処理できるとも仮定しない。今回は独立した通常queryを維持する。

## 検証と再現

Wasmは`0x346015410776174c57cbe72ef6c50906d2f144d9b002e4937e70f8824bf9ed28`、packは`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`。Candle有効のdecision-engineの6 tests、Pythonの76 testsを通過した。Python全体の検証はtorchを含む`.venv`で実行した。実canisterでは不正な状態、0/17/u32最大の段数、非ownerを拒否し、同じqueryの再試行で同じ状態を返し、queryによって既存のupdate jobが変わらないことを確認した。

通常入力の測定（ローカル生成物: `artifacts/query_conditions/measurements.json`）、短い入力・最大選択肢・noul（ローカル生成物: `artifacts/query_conditions/boundaries.json`）、codecの比較（ローカル生成物: `artifacts/query_conditions/codecs-128.json`）、直接query（ローカル生成物: `artifacts/query_conditions/short-direct.json`）、実際の上限時再試行（ローカル生成物: `artifacts/query_conditions/limit-fallback.json`）、protocol検証（ローカル生成物: `artifacts/query_conditions/protocol.json`）を保存した。

```bash
python3 tools/explore_query_conditions.py \
  --output build/query-conditions-new.json
python3 tools/explore_query_conditions.py \
  --case seven-markers-16 --case seven-markers-128 --case noul \
  --output build/query-conditions-boundaries-new.json
python3 tools/explore_lossless_state_codecs.py --output build/query-codecs-new.json
```

最初のツールは専用のlocal canisterに既にモデルがロードされていることを前提とする。必要な場合だけ`--warmup`を付ける。モデルは再uploadしない。codec用ツールは保存済みの128-token分割条件とmodule hashへ結び付けた追加実験で、指定した新しい出力先へ書き出す。
