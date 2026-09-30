use ic_laya_core::{engine::InferenceBackend, Error, TokenInput};
use laya_candle::{continuation::MAX_CONTINUATION_BYTES, pack};
use std::path::PathBuf;

fn fixture(name: &str) -> (laya_candle::LayaModel, TokenInput) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name);
    (
        pack::load_directory(&path).unwrap(),
        serde_json::from_slice(&std::fs::read(path.join("input.json")).unwrap()).unwrap(),
    )
}

#[test]
fn every_boundary_roundtrips_losslessly_and_matches_direct_logits() {
    for name in ["tiny-int8-prenorm", "tiny-int8-postnorm"] {
        let (mut model, input) = fixture(name);
        for qtype in 0..3 {
            let mut input = input.clone();
            input.qtype_id = qtype;
            let expected = model.infer(&input).unwrap();
            for width in [1, 2] {
                let mut session = model.begin_inference(input.clone()).unwrap();
                loop {
                    let bytes = model.export_inference(&session).unwrap();
                    let restored = model.import_inference(&bytes).unwrap();
                    assert_eq!(model.export_inference(&restored).unwrap(), bytes);
                    session = restored;
                    let mut done = None;
                    for _ in 0..width {
                        done = model.step_inference(&mut session).unwrap();
                        if done.is_some() {
                            break;
                        }
                    }
                    if let Some(actual) = done {
                        assert_eq!(actual, expected, "{name} qtype={qtype} width={width}");
                        assert!(matches!(
                            model.export_inference(&session),
                            Err(Error::Transition)
                        ));
                        break;
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_continuations_are_rejected_before_tensor_restore() {
    let (model, input) = fixture("tiny-int8-prenorm");
    let state = model
        .export_inference(&model.begin_inference(input.clone()).unwrap())
        .unwrap();
    for length in [0, 4, 59, state.len() - 1] {
        assert!(model.import_inference(&state[..length]).is_err());
    }
    let mut extra = state.clone();
    extra.push(0);
    assert!(model.import_inference(&extra).is_err());
    assert!(matches!(
        model.import_inference(&vec![0; MAX_CONTINUATION_BYTES + 1]),
        Err(Error::TooLong)
    ));
    for offset in [0, 4, 8, 12, 44, 48, 52, 56, 60] {
        let mut broken = state.clone();
        broken[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(model.import_inference(&broken).is_err(), "offset {offset}");
    }
    let hidden_start = 60 + (input.input_ids.len() + input.markers.len()) * 4;
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut broken = state.clone();
        broken[hidden_start..hidden_start + 4].copy_from_slice(&value.to_le_bytes());
        assert!(matches!(
            model.import_inference(&broken),
            Err(Error::Numeric)
        ));
    }
    let mut different = model;
    different.bundle[0] ^= 1;
    assert!(matches!(
        different.import_inference(&state),
        Err(Error::BindingMismatch)
    ));
}

#[test]
fn phase_shape_and_marker_rows_are_enforced() {
    let (model, input) = fixture("tiny-int8-postnorm");
    let mut session = model.begin_inference(input.clone()).unwrap();
    let initial = model.export_inference(&session).unwrap();
    while session.completed_steps() < model.inference_steps() - 1 {
        model.step_inference(&mut session).unwrap();
    }
    let final_state = model.export_inference(&session).unwrap();
    assert_eq!(
        final_state.len(),
        60 + 4
            * (input.input_ids.len()
                + input.markers.len()
                + input.markers.len() * model.config.hidden_size)
    );
    let mut wrong_phase = initial;
    wrong_phase[44..48].copy_from_slice(&((model.inference_steps() - 1) as u32).to_le_bytes());
    assert!(model.import_inference(&wrong_phase).is_err());
    let mut wrong_marker = final_state;
    let offset = 60 + input.input_ids.len() * 4;
    wrong_marker[offset..offset + 4].copy_from_slice(&0u32.to_le_bytes());
    assert!(model.import_inference(&wrong_marker).is_err());
}
