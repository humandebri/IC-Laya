# INT8変換コストと入力コピーの削減

2026-09-30、クライアント保持の分割queryを実装したブランチで、実モデルの量子化・入力コピーの命令数をlocal IC上で測定した。F32→INT8変換は各Linearで発生するが、今回の128-token入力では総命令数の約1.94%だった。変換だけを高速化しても、大幅な推論高速化にはつながりにくい。

## 変更前の内訳

| 処理 | 128 tokens | 28 tokens |
|---|---:|---:|
| 入力F32の取り出し・コピー `int8.input` | 0.223% | 0.315% |
| 動的量子化 `int8.quantize` | 1.941% | 1.983% |
| INT8行列積・F32書き戻し `int8.matmul` | 76.726% | 84.842% |
| 出力Tensorの構築 `int8.tensor` | 0.00061% | 0.00256% |

量子化にはINT8入力配列とscale配列の確保も含まれる。行列積には出力配列の確保、INT32結果からF32への変換、scaleの適用、必要な場合の末尾paddingも含まれる。したがって「変換全体が1.94%」「整数の積和だけが76.7%」とは解釈できない。配列確保だけの割合は今回の区間計測では分離していない。

計測区間は互いに重ならないLinearの4区間を集計した。分母は開始Embeddingと32ステップを含む推論命令数。開始Embeddingの詳細内訳、Candid処理、クライアントでのコピー・通信はこの内訳に含めていない。詳細計測には既存のowner専用`profile_token_step` update APIを使用した。分割queryのAPIにも同じLinear実装が使われるが、これらの数値は通信を含むquery所要時間の測定ではない。

## 採用した変更

従来は各Linearで`flatten_all()?.to_vec1::<f32>()`によって入力F32をコピーしていた。連続したCPUのF32 Tensorでは、読み取りロックを保持して元の配列を借用し、そのまま量子化するようにした。narrowで生じる先頭オフセットも反映する。非連続配列や他のstorageでは従来の取り出し処理を使う。

量子化の丸め、整数の積和、F32 scaleの適用順序は維持した。出力の`Tensor::from_vec`はCPUではVecの所有権を移すため、もともとF32配列全体のコピーを必要としない。INT8入力・scale・出力配列の確保は引き続き発生する。

| 指標 | 128 tokens | 28 tokens |
|---|---:|---:|
| 変更前の総命令数 | 39,581,457,580 | 8,639,458,280 |
| 変更後の総命令数 | 39,507,510,496 | 8,602,787,488 |
| 観測した削減率 | 0.187% | 0.424% |
| 入力取り出しの命令数、変更前→変更後 | 88,208,828 → 77,340 | 27,225,998 → 57,340 |
| 省いたF32入力コピーの合計 | 85,950,464 bytes | 18,878,464 bytes |
| 省いた1回の入力コピーの最大値 | 2,097,152 bytes | 458,752 bytes |
| logits | 完全一致 | 完全一致 |

いずれも122回のLinearを含む入力ごとの1回の詳細計測。省いたバイト数はLinear間の累計で、同時に減ったメモリ量ではない。総命令数にはallocatorの状態などによる他区間の差もあり、削減率を全入力共通の保証値やwall timeの改善率として扱わない。入力取り出し区間そのものは両入力で99.7%以上減った。

全体の計算コストをさらに削減するなら、行列積やattentionを中心に調べる必要がある。中間状態までINT8にすると、norm・attention・残差加算と量子化誤差の扱いが変わるため、今回のコピー削減とは別の数値設計・品質評価が必要になる。

## 検証と再現

オフセット付き連続Tensor、転置した非連続Tensor、非有限値の拒否を追加テストで確認した。既存の整数参照テスト、PyTorch参照との比較、各phaseのcontinuation往復も実行した。Rustは23 tests、Pythonは72 testsを通過した。

計測結果は[128-token比較](../artifacts/int8_conversion_cost/comparison-128.json)、[28-token比較](../artifacts/int8_conversion_cost/comparison-28.json)に保存した。変更前Wasmは`0x4c3dd516ce2215d5130fbf9b41751e026a2db7ed2f68f3014f6a147be710c038`、変更後は`0xbf0ce7bbbb0afe896ed03ebb0d546407635858a6f8e54ae0a8174387ec7d24aa`。両方でpackは`bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092`を使用した。

モデルをwarmupした専用ローカルプロジェクトでの計測例:

```bash
python3 tools/canister_infer.py \
  --project-root build/client-query-project \
  --identity ic-laya-query-local-test \
  --input artifacts/laya-choice-128-input.json \
  --profile --output build/conversion-profile.json
```

分割queryは回帰検証の記録（ローカル生成物: `artifacts/int8_conversion_cost/query-validation/validation.json`）に別途保存した。従来の[分割query計測](CLIENT_HELD_QUERY_INFERENCE.md)は変更前Wasmの記録として残している。
