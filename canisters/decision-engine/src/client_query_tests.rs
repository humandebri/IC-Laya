use super::*;
use ic_laya_core::engine::InferenceBackend;

#[test]
fn bounded_decoders_keep_the_typed_candid_contract() {
    let interface = candid_interface();
    assert!(interface.contains("benchmark_q8_epilogue : (blob) ->"));
    assert!(interface.contains("begin_token_inference_query : (TokenInput) ->"));
    assert!(interface.contains("continue_token_inference_query : (blob, nat32) ->"));
    assert!(interface.contains("begin_token_inference_int8_query : (TokenInput) ->"));
    assert!(interface.contains("continue_token_inference_int8_query : (blob, nat32) ->"));
    assert!(interface.contains("profile_token_inference_int8_query : (blob, nat32) ->"));
    assert!(interface.contains("begin_token_inference_compressed_query : (TokenInput) ->"));
    assert!(interface.contains("continue_token_inference_compressed_query : (blob, nat32) ->"));
    assert!(interface.contains("begin_token_inference_int8_compressed_query : (TokenInput) ->"));
    assert!(interface.contains("continue_token_inference_int8_compressed_query : (blob, nat32) ->"));
    for name in ["begin_token_inference_batch_query", "begin_token_inference_int8_batch_query", "begin_token_inference_batch_compressed_query", "begin_token_inference_int8_batch_compressed_query"] {
        assert!(interface.contains(&format!("{name} : (TokenInput, nat32) ->")));
    }
}

#[test]
fn fused_begin_is_exact_and_bounded_in_both_engines() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-int8-prenorm");
    let model = laya_candle::pack::load_directory(&dir).unwrap();
    let input: TokenInput = serde_json::from_slice(&std::fs::read(dir.join("input.json")).unwrap()).unwrap();
    for int8 in [false, true] {
        let initial = if int8 { query_begin_int8(&model,input.clone(),||0) } else { query_begin(&model,input.clone(),||0) }.unwrap();
        let QueryInferenceProgress::Continue { state, .. } = initial else { panic!() };
        for width in [1,3,16] {
            let expected = if int8 { query_continue_int8(&model,&state,width,||0) } else { query_continue(&model,&state,width,||0) }.unwrap();
            let fused = query_begin_batch(&model,input.clone(),width,int8,||0).unwrap();
            assert_eq!(fused,expected);
            let packed = transported_query(&model,Some(input.clone()),&[],width,int8,||0).unwrap();
            match (packed,&expected) {
                (QueryInferenceProgress::Continue { state, completed, .. }, QueryInferenceProgress::Continue { state: raw, completed: step, .. }) => {
                    assert_eq!(completed,*step);
                    assert_eq!(query_transport::unpack(&state).unwrap().as_ref(),raw);
                },
                (done,_) => assert_eq!(done,expected),
            }
        }
        for width in [0,17,u32::MAX] {
            assert!(query_begin_batch(&model,input.clone(),width,int8,||0).is_err());
        }
    }
}

#[test]
fn transported_queries_preserve_exact_states_and_logits() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-int8-prenorm");
    let model = laya_candle::pack::load_directory(&dir).unwrap();
    let input: TokenInput = serde_json::from_slice(&std::fs::read(dir.join("input.json")).unwrap()).unwrap();
    for int8 in [false, true] {
        let mut progress = transported_query(&model, Some(input.clone()), &[], 0, int8, || 0).unwrap();
        loop {
            let QueryInferenceProgress::Continue { ref state, .. } = progress else { break };
            let raw = query_transport::unpack(state).unwrap();
            let expected = if int8 { query_continue_int8(&model, &raw, 3, || 0) } else { query_continue(&model, &raw, 3, || 0) }.unwrap();
            let next = transported_query(&model, None, state, 3, int8, || 0).unwrap();
            let repeated = transported_query(&model, None, state, 3, int8, || 0).unwrap();
            assert_eq!(next, repeated);
            match (&next, &expected) {
                (QueryInferenceProgress::Continue { state, completed, .. }, QueryInferenceProgress::Continue { state: plain, completed: expected_step, .. }) => {
                    assert_eq!(completed, expected_step);
                    assert_eq!(query_transport::unpack(state).unwrap().as_ref(), plain);
                },
                _ => assert_eq!(next, expected),
            }
            assert!(transported_query(&model, None, state, 3, !int8, || 0).is_err());
            progress = next;
        }
    }
}

#[test]
fn quantized_queries_retry_exactly_and_reject_f32_states() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/tiny-int8-prenorm");
    let model = laya_candle::pack::load_directory(&dir).unwrap();
    let input: TokenInput =
        serde_json::from_slice(&std::fs::read(dir.join("input.json")).unwrap()).unwrap();
    let mut first = query_begin_int8(&model, input.clone(), || 10).unwrap();
    let mut second_input = input.clone();
    second_input.qtype_id = 1;
    let mut second = query_begin_int8(&model, second_input, || 10).unwrap();
    let QueryInferenceProgress::Continue {
        state: f32_state, ..
    } = query_begin(&model, input, || 10).unwrap()
    else {
        panic!()
    };
    assert!(query_continue_int8(&model, &f32_state, 1, || 10).is_err());
    loop {
        let mut remaining = false;
        for progress in [&mut first, &mut second] {
            if let QueryInferenceProgress::Continue { state, .. } = progress {
                assert!(query_continue(&model, state, 1, || 10).is_err());
                for limit in [0, 17, u32::MAX] {
                    assert!(query_continue_int8(&model, state, limit, || 10).is_err());
                }
                let next = query_continue_int8(&model, state, 2, || 10).unwrap();
                assert_eq!(next, query_continue_int8(&model, state, 2, || 10).unwrap());
                *progress = next;
                remaining = true;
            }
        }
        if !remaining {
            break;
        }
    }
    for progress in [first, second] {
        let QueryInferenceProgress::Done {
            logits,
            completed,
            total,
            ..
        } = progress
        else {
            panic!()
        };
        assert_eq!(completed, total);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn query_states_are_independent_retryable_and_match_direct_inference() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/tiny-int8-prenorm");
    let mut model = laya_candle::pack::load_directory(&dir).unwrap();
    let input: TokenInput =
        serde_json::from_slice(&std::fs::read(dir.join("input.json")).unwrap()).unwrap();
    let mut second = input.clone();
    second.qtype_id = 1;
    let expected = [model.infer(&input).unwrap(), model.infer(&second).unwrap()];
    let mut states = [
        query_begin(&model, input, || 10).unwrap(),
        query_begin(&model, second, || 10).unwrap(),
    ];
    let mut done = [false; 2];
    while !done.iter().all(|v| *v) {
        for i in 0..2 {
            if done[i] {
                continue;
            }
            let QueryInferenceProgress::Continue { state, .. } = &states[i] else {
                panic!("missing state")
            };
            for bad_steps in [0, 17, u32::MAX] {
                assert!(matches!(
                    query_continue(&model, state, bad_steps, || 10),
                    Err(Error::Invalid(_))
                ));
            }
            let width = if i == 0 { 1 } else { 16 };
            let next = query_continue(&model, state, width, || 10).unwrap();
            assert_eq!(
                next,
                query_continue(&model, state, width, || 10).unwrap()
            );
            if let QueryInferenceProgress::Done {
                logits,
                completed,
                total,
                ..
            } = &next
            {
                assert_eq!(logits, &expected[i]);
                assert_eq!(completed, total);
                done[i] = true;
            }
            states[i] = next;
        }
    }
    // The public APIs deliberately reject uncalibrated packs.
    assert!(matches!(
        check_query_pack(&model),
        Err(Error::BindingMismatch)
    ));
}
