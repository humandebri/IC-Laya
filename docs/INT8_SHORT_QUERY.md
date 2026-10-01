# Laya raw queryの16-token境界

2026-09-25に、現行packのtokenizerとlocal canisterで短い入力を確認した。[入力とupdate実測](../artifacts/int8_optimization_v4/short-inputs.json)、同じWasmでのupdate/query比較（ローカル生成物: `artifacts/int8_optimization_v4/short-query-results.json`）にtoken ID、Wasm・pack hash、logits、命令数を保存した。

現行packのowner専用`infer_tokens_query`は、タスクの文章やschema名に関係なく**合計16 tokensまで**受け付ける。17 tokens以上は推論前に`TooLong`を返す。通常のupdate推論にはこの短入力制限を加えていない。

以前のChoice測定で使った28 tokensは、そのschemaのprefixが26 tokensで、本文の最小1 tokenと末尾`[SEP]`を加えた長さだった。モデルやcanisterの下限ではない。新しいタスクではそのschemaを前提にする必要はない。

新しい短いschemaなら28未満の入力を構成できる。現行の生成規則（`choice/noul question: ...`、選択肢ごとの`[MASK]`、先頭空白付き選択肢、本文、区切り）に沿ってtokenizeした例:

| 短い入力 | tokens | 同じWasmのupdate | 実query | 結果 |
|---|---:|---:|---:|---|
| Choice「Choose.」/ red・blue・green / state「blue」 | 16 | 4,728,422,785命令 | 4,724,809,977命令 | 成功、logits完全一致 |
| Choice「Color?」/ 同じ選択肢・state | 16 | 4,728,014,426命令 | 4,725,552,358命令 | 成功、logits完全一致 |
| Choice「Pick a color.」/ 同じ選択肢・state | 18 | 5,501,254,555命令 | 推論前に拒否 | `TooLong` |
| Noul「Safe?」/ no・yes / state「yes」 | 15 | 4,680,194,671命令 | 4,676,607,838命令 | 成功、logits完全一致 |
| Noul「Is this safe?」/ 同じ選択肢・state | 17 | 5,156,389,623命令 | 推論前に拒否 | `TooLong` |

上表のupdate命令数は16-tokenガード付きWasmで取り直した値。query側はowner専用の`infer_tokens_query`でraw logitsを返す。非ownerは拒否される。結果は認証付きのReceiptではなく、認証・改ざん検証済みのquery応答でもない。従来の`evaluate`やexecutor経路はupdateのまま。

## タスク非依存のquery受付境界

現行packで、15・16・17 tokens、マーカー2〜7個、qtype全3種を組み合わせた54条件を[updateで測定](../artifacts/int8_optimization_v4/query-limit-grid.json)した。16 tokensの最大は**4,757,429,998命令**、17 tokensの最小は**5,157,045,105命令**。ガード追加後の実queryでも16 tokensの18条件がすべて成功し、最大**4,756,310,000命令**だった。17 tokensの18条件と128 tokensはすべて推論前に`TooLong`となり、17-token入力はupdateでは成功した（ガード検証（ローカル生成物: `artifacts/int8_optimization_v4/query-guard-check.json`））。したがって現行packに対するquery受付は**最大16 tokens、17 tokens以上は拒否**とした。別packへの交換時はガードが`BindingMismatch`で拒否し、別途再測定を求める。

この境界は測定したWasm・packに対するもので、全ての将来のコードやモデルに対する数学的上界ではない。短入力の選択肢マーカー数やqtypeをまたぐ実測の最悪値には約244M命令の余裕がある。ガード付きWasmでも元の96入力との互換性（ローカル生成物: `artifacts/int8_optimization_v4/guarded-query-vs-baseline-canister.json`）と、128-token通常updateの継続（ローカル生成物: `artifacts/int8_optimization_v4/guarded-query-update-limits.json`）を確認した。

**短縮の主な懸念は意味の欠落である。** 「Choose.」「Safe?」は費用測定には使えても、元の質問文や判断条件を表せない。「Color?」はこの例の選択肢を指すが、一般の判断ルールを表す余地は小さい。実際、同じstate「yes」と選択肢no・yesでも、質問「Safe?」はyes、「Is this safe?」はnoを選んだ。どちらが正しいかは、この例だけでは判定できない。schemaのinstructionsやoptionsを変えるとschema hashが変わるため、既存の登録をそのまま流用せず、versionを更新して再登録する必要がある。本文を短くすると判断材料も減る。これら5入力には正解ラベルや品質評価がなく、短入力の正答率は不明。queryを使える長さはschema、本文、選択肢数、実際のWasm次第なので、実運用では入力ごとの予算超過時にupdateへ切り替える設計が必要になる。
