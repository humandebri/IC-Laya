use ic_laya_core::TokenInput;
use laya_candle::pack;
use std::path::PathBuf;

#[test]
fn q8_continuation_roundtrips_every_phase_for_both_head_layouts() {
    for name in ["tiny-int8-prenorm", "tiny-int8-postnorm"] {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures")
            .join(name);
        let model = pack::load_directory(&dir).unwrap();
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        for case in cases.as_array().unwrap() {
            let input: TokenInput = serde_json::from_value(case["input"].clone()).unwrap();
            let mut direct = model.begin_quantized(input.clone()).unwrap();
            let mut resumed = model.begin_quantized(input).unwrap();
            loop {
                let bytes = model.export_quantized(&resumed).unwrap();
                assert_eq!(&bytes[..4], b"LAYI");
                assert!(model.import_inference(&bytes).is_err());
                resumed = model.import_quantized(&bytes).unwrap();
                assert_eq!(model.export_quantized(&resumed).unwrap(), bytes);
                let a = model.step_quantized(&mut direct).unwrap();
                let b = model.step_quantized(&mut resumed).unwrap();
                assert_eq!(a, b);
                if let Some(logits) = a {
                    assert_eq!(
                        logits.len(),
                        case["input"]["markers"].as_array().unwrap().len()
                    );
                    assert!(logits.iter().all(|v| v.is_finite()));
                    assert!(model.export_quantized(&direct).is_err());
                    break;
                }
            }
        }
    }
}

#[test]
fn q8_rejects_wrong_binding_shape_and_scales() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-int8-prenorm");
    let model = pack::load_directory(&dir).unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
    let input: TokenInput = serde_json::from_value(cases[0]["input"].clone()).unwrap();
    let scale_offset = 60 + 4 * (input.input_ids.len() + input.markers.len());
    let state = model.begin_quantized(input.clone()).unwrap();
    let good = model.export_quantized(&state).unwrap();
    let mut old_revision = good.clone();
    old_revision[8..12].copy_from_slice(&1u32.to_le_bytes());
    assert!(model.import_quantized(&old_revision).is_err());
    let f32 = model
        .export_inference(&model.begin_inference(input).unwrap())
        .unwrap();
    assert!(model.import_quantized(&f32).is_err());
    for offset in [0, 4, 8, 12, 44, 48, 52, 56, 60] {
        let mut bad = good.clone();
        bad[offset..offset + 4].fill(255);
        assert!(model.import_quantized(&bad).is_err(), "offset {offset}");
    }
    for scale in [0., -1., f32::NAN, f32::INFINITY, f32::MAX] {
        let mut bad = good.clone();
        bad[scale_offset..scale_offset + 4].copy_from_slice(&scale.to_le_bytes());
        assert!(model.import_quantized(&bad).is_err());
    }
    assert!(model.import_quantized(&good[..good.len() - 1]).is_err());
    let mut bad = good.clone();
    bad.push(0);
    assert!(model.import_quantized(&bad).is_err());
}
