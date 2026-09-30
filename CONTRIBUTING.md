# Contributing to IC-Laya

Issues and pull requests are welcome. Describe the behavior you observed, the expected behavior, the Rust toolchain version, and the commands needed to reproduce it. For inference or performance changes, include the model pack identity, input length, Wasm hash, and the measurement method when available.

Before opening a pull request, run the checks relevant to your change:

```bash
cargo test --workspace --locked
cargo check -p decision-engine --features candle --locked
python3 -m unittest discover -s tests -v
```

For canister changes, also build the affected Wasm module with `tools/build_one.sh`. The [CI workflow](.github/workflows/ci.yml) runs native tests and all three Wasm builds. If a check needs a local model or IC replica that you cannot provide, state that in the pull request.

Do not commit pretrained weights, downloaded checkpoints, private keys, credentials, local canister state, or generated build output. The checked-in `fixtures/` contain random test weights. Keep measurements and compatibility claims tied to the exact input, model pack, and Wasm used.

Generated files in `artifacts/` are ignored by default. The explicit allowlist in `.gitignore` retains fixed inputs, compact reference logits, final summaries, and explanatory Markdown. Keep detailed query traces, profiles, repeated measurements, downloaded proposal snapshots, and experimental source copies local; do not force-add them. Compact summaries should identify their inputs, model, Wasm, and source measurements by hash. Put ad hoc results in `artifacts/local/` or `build/`. The small `fixtures/tiny-*/model.bin` files remain versioned test inputs.

By contributing, you agree that your contribution is licensed under the repository's [MIT License](LICENSE). Third-party code, models, and data require their own redistribution rights and attribution.
