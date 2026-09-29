# Run Laya locally

This guide takes a fixed Laya checkpoint through F32 export, INT8 conversion, and inference in a local Internet Computer canister. Run commands from the repository root. No pretrained weights are distributed with IC-Laya.

## Prerequisites

- Git and Rust installed through rustup. The repository pins Rust in `rust-toolchain.toml` and requests the Wasm target.
- Python 3.12 and a virtual environment.
- ICP CLI (`icp`). The commands below were checked with version 1.0.2. See the [official CLI documentation](https://cli.internetcomputer.org/1.0/).
- Several GB of available RAM and disk space for dependencies, builds, and model conversion. The F32 model data is about 1.68 GB and INT8 model data about 423 MB; conversion also uses temporary memory.

```bash
git clone https://github.com/humandebri/IC-Laya.git
cd IC-Laya
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

For a small test without a pretrained checkpoint:

```bash
cargo run --locked -p laya-candle --bin laya-infer -- \
  fixtures/tiny-prenorm fixtures/tiny-prenorm/input.json
```

The output identifies the backend as `SyntheticFixture`. These random weights test the inference path, not language understanding.

## Obtain the checkpoint

Review the model's license and terms before downloading it. The conversion bridge targets `convaiinnovations/laya-typed-decisions` at revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`; it is not a general converter for arbitrary Laya revisions.

```bash
mkdir -p checkpoints/laya-source
curl --fail --location --retry 3 \
  'https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/f9ab0b228f0fc0f14d873dbc99038f135c2da1b2/model.safetensors' \
  --output checkpoints/laya-source/model.safetensors
```

An existing copy of that exact revision's `model.safetensors` can be used instead. The file used in the setup validation has SHA-256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`. `checkpoints/` is ignored by Git.

## Export and quantize

```bash
.venv/bin/python tools/laya_port_bridge.py export \
  --out checkpoints/laya-port \
  --header checkpoints/laya-source/model.safetensors \
  --weights checkpoints/laya-source/model.safetensors

.venv/bin/python tools/quantize_pack.py \
  checkpoints/laya-port/pack checkpoints/laya-int8
```

The bridge reads the local weight header and fetches configuration and tokenizer JSON from the fixed upstream revision. It writes the F32 pack to `checkpoints/laya-port/pack`. The quantizer writes the INT8 pack to `checkpoints/laya-int8`. Both pack outputs must be new directories; choose new output paths for a repeat export.

The bridge reports missing tensors and shape mismatches. Its `parity_verified: false` means this command does not run numerical parity checks. Separate, limited comparisons are recorded in [INT8.md](INT8.md). The port computes option logits and omits the upstream act/escalation head.

## Build and start a local canister

```bash
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine
```

This creates `build/decision-engine.wasm` and `build/decision-engine.did`. It does not embed the checkpoint.

Create a dedicated local identity **before starting the network**, so the CLI seeds its local test balance. The following plaintext identity is for local development only; use a different name if it already exists. The seed is written into the ignored build directory.

```bash
icp identity new ic-laya-local-guide --storage plaintext \
  --output-seed build/local-guide-seed.txt
icp network start local -d
icp network status local --json
icp cycles balance -e local --identity ic-laya-local-guide
```

If port 8000 is occupied, add this network entry to `icp.yaml` before starting, using a free port. With the tested CLI, `gateway` belongs to the network entry, not the top level of `icp.yaml`. See the [configuration reference](https://cli.internetcomputer.org/1.0/reference/configuration/).

```yaml
networks:
  - name: local
    mode: managed
    gateway:
      port: 8001
```

Create a fresh canister with enough local cycles for model upload and warmup:

```bash
icp canister create decision-engine --cycles 7t \
  -e local --identity ic-laya-local-guide

.venv/bin/python tools/canister_infer.py --initialize \
  --identity ic-laya-local-guide \
  --pack checkpoints/laya-int8 --stepped \
  --input artifacts/laya-noul-input.json \
  --output build/first-inference.json
```

`--initialize` installs into the fresh, empty canister. It does not reinstall an existing module. The script uploads the model and tokenizer in chunks, warms the model, and returns raw logits and instruction counts. The example input is a checked-in token sequence for the fixed checkpoint; it does not require a separate tokenizer installation.

## Run another input

Keep the same owner identity and omit `--initialize` and `--pack` to reuse the loaded model:

```bash
.venv/bin/python tools/canister_infer.py \
  --identity ic-laya-local-guide --stepped \
  --input artifacts/laya-choice-128-input.json \
  --output build/choice-128.json
```

`--stepped` runs resumable inference, grouping up to 16 model phases per update. Omit it to try a single update. The input limit is 128 tokens including the schema prefix; a single update is not guaranteed for every input or build.

The owner-only `infer_tokens_query` API accepts at most 16 tokens and checks a fixed pack hash. The optional `--max-update-instructions` route also checks fixed Wasm and pack hashes. A regenerated pack or rebuilt Wasm may differ, so these measured limits are not automatic guarantees for a new build. See [query constraints](INT8_SHORT_QUERY.md) and [instruction budgeting](INT8_INSTRUCTION_BUDGET.md).

## Troubleshooting and stopping

- **Insufficient local funds:** create the identity before the first network start. If it was created later, stop and restart only this project's local network, then check its local balance. See [local development](https://cli.internetcomputer.org/1.0/guides/local-development/). Allow the previous process time to release its port before restarting.
- **Frozen canister or insufficient cycles during upload:** check `icp canister status decision-engine -e local --identity ic-laya-local-guide`. Add local cycles with `icp canister top-up decision-engine --amount 5t -e local --identity ic-laya-local-guide` and retry with `--pack`, without `--initialize` if installation already succeeded. Uploading again replaces the active model.
- **Already installed:** omit `--initialize`. For a deliberate upgrade, preserve the owner and use the CLI's upgrade mode; after upgrading, run the inference script with `--warmup` to rebuild the model from stable bytes. In-progress inference jobs do not survive upgrades.
- **BindingMismatch in query or budget routing:** the measured hash does not match. Use ordinary update or stepped inference; do not remove the check to reuse an old performance claim.

```bash
icp network stop local
```

This stops the network for the current project. Model files, identity, and local state remain on disk. This guide deploys only to a local network and does not enable real-fund transfers.

## Validation record

The [setup validation record](../artifacts/oss_setup_validation.json) records a fresh managed checkout, Python environment, model export, Wasm build, and fresh local canister run. Both 45-token and 128-token inputs completed in two updates. The fixed-revision checkpoint was reused from a local copy; the full weight download was not repeated. Its URL returned HTTP 200, and configuration and tokenizer files were fetched again. This verifies the setup path, not decision accuracy.
