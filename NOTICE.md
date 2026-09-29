# Model and dependency notice

The MIT license applies to the new source in this package, not to third-party
libraries or pretrained checkpoints. Candle, tokenizers, Rust CDK, PyTorch,
NumPy, safetensors and other dependencies retain their respective licenses.

No pretrained Laya/ModernBERT weights are included. The small binary files in
`fixtures/` contain newly generated random test tensors. They are not trained
models and are not a substitute for task evaluation.

IC-Laya is an independent implementation, not an official Convai,
TypeSafe/Jev, Hugging Face, or DFINITY release. The inference implementation
was written against explicit tensor/mathematical contracts and public model
architecture documentation. Limited F32 and INT8 comparisons against the
upstream checkpoint are recorded in `docs/INT8.md`; broader equivalence and
task-specific accuracy have not been established.
