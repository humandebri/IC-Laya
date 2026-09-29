# IC-Laya

IC-Laya is an independent Rust and Internet Computer (ICP) canister implementation of Laya's typed decisions: Choice, Noul, and Score. It includes F32 and W8A8 INT8 inference, a tokenizer adapter, canisters, a mock workflow, and verification tools.

**Pretrained weights are not included.** The small models in `fixtures/` contain random test weights. They do not demonstrate language understanding or decision accuracy. To run the real model, provide a checkpoint and tokenizer whose licenses permit your intended use.

## What it does

- Convert a Laya checkpoint into a model pack and run inference in native Rust or a local IC canister.
- Run owner-only raw inference by update for inputs up to 128 tokens, including the schema prefix. A measured input completed in one update; a resumable path is also available.
- Run an owner-only raw query for up to 16 tokens with the measured Wasm and INT8 pack. Inputs of 17 or more tokens, and other packs, are rejected before inference.
- Exercise typed results and workflows locally with a mock ledger.

The raw inference APIs return logits. Resumable inference is not connected to the current `evaluate` or executor path. Real-fund transfers are disabled: `LimitedLive` returns `LiveDisabled`.

## Quick start

The Rust version is pinned in [`rust-toolchain.toml`](rust-toolchain.toml). Python 3 is needed for the Python tools.

```bash
cargo test --workspace --locked
cargo check -p decision-engine --features candle --locked
cargo run --locked -p laya-candle --bin laya-infer -- \
  fixtures/tiny-prenorm fixtures/tiny-prenorm/input.json
```

The last command checks the inference path with random test weights; it does not measure decision quality. Run the Python reference tests with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

With the `wasm32-unknown-unknown` Rust target installed, build the canister Wasm modules and Candid interfaces with:

```bash
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine
bash tools/build_one.sh executor
bash tools/build_one.sh mock-ledger
```

Outputs go to `build/`; these commands do not deploy a canister. Follow the English [local inference guide](docs/GETTING_STARTED.md) for checkpoint download, model pack conversion, and local canister inference.

## What has been measured

| Check | Result | Scope |
| --- | --- | --- |
| Real checkpoint versus the upstream implementation | Maximum absolute logit difference: 4.89e-6 for F32 and 0.141 for INT8; argmax agrees on 4/4 inputs | Four fixed inputs. [Details](docs/INT8.md) |
| 128-token INT8 inference in a local canister | One update used 39.248B instructions; resumable inference completed in two updates | A repeated Choice input with a fixed Wasm and pack. [Measurements](docs/INT8_OPTIMIZATION_V4.md) |
| Short raw queries | All 18 tested 16-token cases succeeded; maximum was 4.756B instructions. Inputs of 17 or more tokens are rejected | Owner-only, fixed Wasm and pack. [Measurements](docs/INT8_SHORT_QUERY.md) |
| Handwritten English classification examples | 14 of 16 matched their assigned labels | A small, unrepresentative probe, not an accuracy estimate. [Inputs and results](docs/INT8_PRACTICAL_128.md) |

These results come from limited inputs in a local environment. They do not guarantee that every 128-token input fits in one update or establish accuracy, calibration, or safety for real tasks. Security or financial decisions require evaluation on the intended use case and human review. The query API returns raw logits, not an authenticated Receipt or a certified response.

## Repository layout

| Path | Contents |
| --- | --- |
| `crates/ic-laya-core/` | Types, schemas, math, and workflows |
| `crates/laya-candle/` | F32 and INT8 inference and model pack loading |
| `crates/hf-tokenizer/` | Tokenizer adapter |
| `canisters/` | Decision engine, executor, and mock ledger |
| `tools/` | Pack conversion, builds, local runs, and benchmarks |
| `fixtures/`, `tests/` | Random-weight fixtures and reference tests |
| `artifacts/`, `docs/` | Recorded outputs, instruction counts, and design notes |

See [INT8 implementation and local setup](docs/INT8.md), [performance measurements](docs/INT8_OPTIMIZATION_V4.md), and [instruction budgeting](docs/INT8_INSTRUCTION_BUDGET.md). The [documentation index](docs/README.md) identifies current guides and historical research notes.

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for validation commands and the policy on model files. Report vulnerabilities through the private channel in [SECURITY.md](SECURITY.md). Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

New source code in this repository is available under the [MIT License](LICENSE). Dependencies and user-provided checkpoints retain their own licenses; see [NOTICE](NOTICE.md). This is not an official release from the upstream model authors or DFINITY.
