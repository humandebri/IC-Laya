# Changelog

## Unreleased

- Accept only INT8 model packs in native and canister inference. F32 matrix loading and inference, the `from_tensors` constructor, and the unused `verdict-candle` backend have been removed.
- Keep F32 vectors/scales, checkpoint conversion, and independent numerical references. Existing INT8 packs remain compatible.
- Generate INT8 packs for sized synthetic benchmarks. Upstream comparison now accepts `--int8` only; the removed `--f32` runtime option is no longer needed.
- Migration: convert any F32 pack with `tools/quantize_pack.py` before loading it. An upgraded canister holding a legacy F32 pack must upload an INT8 pack before warmup; no automatic conversion occurs.

## 0.1.0-alpha.1 — 2026-09-29

Initial experimental source release:

- Independent Rust implementation of Laya option-logit inference with W8A8 INT8 model packs. F32 export and inference are retained for conversion and numerical reference.
- Owner-only local canister inference with direct updates, resumable updates, and a measured 16-token raw query path for a fixed pack.
- Model conversion, benchmark tools, numerical reference tests, and recorded local measurements.
- English setup and contribution guides, private security reporting, and issue/PR templates.

Pretrained weights are not distributed. Real-fund transfers remain disabled. Model parity and decision-quality checks cover limited inputs; this is not a production-readiness or task-accuracy certification. See the [setup guide](docs/GETTING_STARTED.md) and [measurement index](docs/README.md).
