# Laya F32 port: source contract and remaining parity work

> Historical note: this document describes an earlier revision. Statements about missing implementations, tests, or measurements are not the current project status. Start with the [English local inference guide](GETTING_STARTED.md) and [current measurement index](README.md).

## 実装した演算

batch=1・paddingなしのtoken列に対して、embedding/norm、pre-norm ModernBERT、QKV、split-half RoPE、global/local attention、GeGLU、final norm、question-type embedding、decision Transformer、marker gather、LayerNorm→Linear→GELU→Linear scorerを実装した。decision層はpre/post normとReLU/GELUをconfigで切り替える。

これは公開checkpointとの同一性を確認済みの公式portではない。手元のPyTorch参照も同じ設計に基づく独立コードであり、Layaのupstream APIそのものではない。act/escalate headを含めず、option logitsだけを計算する。

## canonical pack

- `manifest.json`: config・tensor位置/shape/hash・tokenizer hash・source identity・qtype順。
- `model.bin`: little-endian F32 tensorを連続配置。最大2GiB、1 tensor最大256MiB。
- `tokenizer.json`: HF tokenizer。raw bytesのSHA-256で識別。
- bundle ID: **manifestのraw bytes**のSHA-256。JSONを整形し直せば別bundleになる。

`config`は`fixtures/tiny-prenorm/manifest.json`を構造の例として参照できるが、値はランダム小型テスト用。実Laya用の値として利用しない。

対応表の文法:

```json
{
  "embeddings.weight": "verified.source.embedding.weight",
  "encoder.0.qkv.weight": {
    "concat": ["verified.q.weight", "verified.k.weight", "verified.v.weight"],
    "axis": 0
  },
  "scorer.dense.weight": {
    "source": "verified.transposed.weight",
    "transpose": true
  }
}
```

これは一部分の文法例。全canonical tensorを含める必要がある。`crates/laya-candle/src/lib.rs::expected_tensors`を正本とし、loader/exporter双方がexact setとshapeを検査する。疑わしいbiasを0で足す、足りないtensorをrandomで補う、不要なものを無言で無視する処理はない。

## 最低限のparity手順

1. source model、tokenizer、config、コードを同じimmutable revisionに固定する。
2. 3 primitiveと2〜7 optionsについて、upstreamの実際のbuild_sequence出力とinput_ids/marker/qtypeを保存する。
3. 元APIが切り詰める入力を、IC版は拒否する。両方の保持された意味が同一な短文だけでparity比較する。
4. 元モデルのraw option logitsを保存する。softmax・temperature・ppm変換の前後を分ける。
5. Candle側の各block境界を比較して、RoPE、local-mask境界、GeGLU順、norm epsilon、head activation、qtype加算位置を特定する。
6. 合成テストが通ってもupstream実装との差は残り得る。最終logitsと判定一致が確認されるまで本番calibrationを作らない。

## canisterへのロード

`begin_upload(manifest, tokenizer_length, special_tokens)` → `upload_chunk`で`model.bin`を順に送信 → その末尾へ`tokenizer.json`を送信 → `start_warmup` → `warmup_next`をtensor数分実行。

1chunk最大1MiB。重複chunkは既存bytesと一致する場合だけ許可。warm-upは1回に1tensorで、SHA/shape/finite検査を行う。全tensor完了後にmodel hashをactiveにする。model入替中は推論不可とし、旧新2モデル同時heap保持を前提にしない。

1tensor206MB等のロードがupdate予算に収まるかは未実測。失敗する場合は分割Tensor構築・embedding部分読み込み・別pack/kernelなどが追加実装になる。F32で421Mなら単純なweightsだけで約1.7GB規模になり、loader一時コピーとactivationは別途必要である。サイズから実行可能と断言しない。

アップグレード後はweights bytesとupload metadataを保持するが、Candle Tensorのheapは再構築する。warm-upまで実モデルは利用不可。カーネル/数値modeの変更はbundleとcalibrationを変更する。

## 未実装

- BF16/F16/INT8/INT4推論と量子化kernel
- 実checkpointの確認済みtensor mapと上流golden logits
- 実際のHF tokenizerでのsequence一致テスト
- training / distillation / RLCD / learned act head
- 初期化時間・warm推論・stable save込みのinstruction計測
- mainnet運用可能なモデル管理GUI

## 実checkpointの調査結果

上流checkpointの実際のtensor名・shape・dtype・special token・校正温度を取得して確認した結果を[MODEL_PORT_FINDINGS.md](MODEL_PORT_FINDINGS.md)に記録した。想定canonical名の大半が実物と異なり、mask token idもfixtureと異なる。draft mappingは`tools/laya_port_bridge.py discover`で再生成できる。
