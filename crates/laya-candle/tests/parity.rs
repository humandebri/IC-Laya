use ic_laya_core::{engine::InferenceBackend,TokenInput,BackendKind,Error};
use std::path::PathBuf;
fn dir(name:&str)->PathBuf{PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(name)}
#[test]
fn int8_logits_match_independent_pytorch_reference(){
    for folder in ["tiny-int8-prenorm","tiny-int8-postnorm"]{
        let d=dir(folder);let mut model=laya_candle::pack::load_directory(&d).unwrap();
        assert_eq!(model.kind(),BackendKind::SyntheticFixture);
        let cases:serde_json::Value=serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
        for case in cases.as_array().unwrap(){
            let input:TokenInput=serde_json::from_value(case["input"].clone()).unwrap();
            let expected:Vec<f32>=serde_json::from_value(case["expected_logits"].clone()).unwrap();
            let output=model.infer(&input).unwrap();assert_eq!(expected.len(),output.len());
            for (a,b) in output.iter().zip(expected.iter()) {assert!((a-b).abs()<=0.002,"{folder}: {a} versus {b}");}
        }
    }
}
#[test]
fn corrupted_tensor_rejected(){
    let d=dir("tiny-int8-prenorm");let raw=std::fs::read(d.join("manifest.json")).unwrap();let mut builder=laya_candle::pack::Builder::new(&raw).unwrap();
    let e=builder.next_entry().unwrap().clone();let all=std::fs::read(d.join("model.bin")).unwrap();
    let mut bytes=all[e.offset as usize..(e.offset+e.length) as usize].to_vec();bytes[0]^=1;assert!(builder.push(&bytes).is_err());
}
#[test]
fn rejects_missing_option_markers(){let mut model=laya_candle::pack::load_directory(&dir("tiny-int8-prenorm")).unwrap();let bad=TokenInput{input_ids:vec![1,5,6,2],markers:vec![1,2],qtype_id:0};assert!(model.infer(&bad).is_err());}
#[test]
fn rejects_unknown_qtype(){let mut model=laya_candle::pack::load_directory(&dir("tiny-int8-prenorm")).unwrap();let bad=TokenInput{input_ids:vec![1,3,3,2],markers:vec![1,2],qtype_id:3};assert!(matches!(model.infer(&bad),Err(Error::Invalid(_))));}

#[test]
fn profiled_inference_matches_plain_inference(){
    // The profiler must be observation-only: identical logits, and the phase
    // costs must account for the whole call.
    use std::cell::Cell;
    for folder in ["tiny-int8-prenorm","tiny-int8-postnorm"]{
        let d=dir(folder);
        let mut plain=laya_candle::pack::load_directory(&d).unwrap();
        let mut profiled=laya_candle::pack::load_directory(&d).unwrap();
        let cases:serde_json::Value=serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
        for case in cases.as_array().unwrap(){
            let input:ic_laya_core::TokenInput=serde_json::from_value(case["input"].clone()).unwrap();
            let expected=plain.infer(&input).unwrap();
            // A counter that advances by a fixed amount per read, so phase deltas
            // are non-zero and their sum is the total.
            let tick=Cell::new(0u64);
            let counter=||{tick.set(tick.get()+10);tick.get()};
            let phases=profiled.infer_profiled(&input,&counter).unwrap();
            let actual=profiled.infer(&input).unwrap();
            assert_eq!(expected,actual,"{folder}: profiled model diverged from plain inference");
            let names:Vec<&str>=phases.iter().map(|p|p.name).collect();
            assert_eq!(names,vec!["embedding","encoder","final_norm","decision","scorer","decode"],"{folder}");
            let total:u64=phases.iter().map(|p|p.instructions).sum();
            assert_eq!(total,60,"{folder}: phases must sum to the measured span");
            assert!(phases.iter().all(|p|p.instructions>0),"{folder}: every phase must consume instructions");
        }
    }
}

#[test]
fn int8_manifest_and_scales_are_validated() {
    use laya_candle::pack::{Builder, Manifest};
    let d = dir("tiny-int8-prenorm");
    let raw = std::fs::read(d.join("manifest.json")).unwrap();
    let all = std::fs::read(d.join("model.bin")).unwrap();
    let mut manifest = Manifest::parse(&raw).unwrap();
    manifest.format = "ic-laya-f32-pack-v1".into();
    assert!(manifest.validate().is_err());
    let mut b = Builder::new(&raw).unwrap();
    while let Some(e) = b.next_entry().cloned() {
        let bytes = &all[e.offset as usize..(e.offset+e.length) as usize];
        if e.storage == laya_candle::pack::Storage::I8Row {
            let mut broken = bytes.to_vec();
            broken[0] ^= 1;
            assert!(b.push(&broken).is_err());
            break;
        }
        b.push(bytes).unwrap();
    }
    for scale in [0f32, -1., f32::NAN, f32::INFINITY] {
        let mut raw = vec![1u8; 17];
        raw.extend(scale.to_le_bytes());
        assert!(laya_candle::int8::Int8Matrix::from_bytes(&raw, 1, 17).is_err());
    }
}

#[test]
fn int8_integer_dot_handles_tail_sign_zero_and_max_width() {
    use candle_core::{Device, Tensor};
    for width in [1, 15, 16, 17, 33, 16384] {
        let mut bytes: Vec<u8> = (0..width).map(|i| if i % 2 == 0 { 127u8 } else { (-128i8) as u8 }).collect();
        bytes.extend(0.25f32.to_le_bytes());
        let w = laya_candle::int8::Int8Matrix::from_bytes(&bytes, 1, width).unwrap();
        let row: Vec<f32> = (0..width).map(|i| if i % 2 == 0 { 127. } else { -127. }).collect();
        let expected: i32 = row.iter().zip(&bytes).map(|(a,b)| *a as i32 * (*b as i8) as i32).sum();
        let mut input = row;
        input.extend(vec![0.; width]);
        let x = Tensor::from_vec(input, (2, width), &Device::Cpu).unwrap();
        assert_eq!(w.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![expected as f32 * 0.25, 0.]);
        let bad = Tensor::from_vec(vec![f32::NAN; width], (1, width), &Device::Cpu).unwrap();
        assert!(w.forward(&bad).is_err());
    }
}

#[test]
fn stepped_inference_is_identical_and_bound_to_the_pack() {
    for suffix in ["int8-prenorm", "int8-postnorm"] {
        let d = dir(&format!("tiny-{suffix}"));
        let mut model = laya_candle::pack::load_directory(&d).unwrap();
        let input: TokenInput = serde_json::from_slice(&std::fs::read(d.join("input.json")).unwrap()).unwrap();
        let expected = model.infer(&input).unwrap();
        let mut session = model.begin_inference(input).unwrap();
        for step in 0..model.inference_steps() {
            assert_eq!(session.completed_steps(), step);
            let result = model.step_inference(&mut session).unwrap();
            if step + 1 == model.inference_steps() { assert_eq!(result.unwrap(), expected); }
            else { assert!(result.is_none()); }
        }
        assert!(matches!(model.step_inference(&mut session), Err(Error::Transition)));
        model.bundle[0] ^= 1;
        assert!(matches!(model.step_inference(&mut session), Err(Error::BindingMismatch)));
    }
}

#[test]
fn detailed_profile_is_observation_only() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TICK: AtomicU64 = AtomicU64::new(0);
    fn counter() -> u64 { TICK.fetch_add(1, Ordering::Relaxed) }
    let d = dir("tiny-int8-prenorm");
    let mut model = laya_candle::pack::load_directory(&d).unwrap();
    let input: TokenInput = serde_json::from_slice(&std::fs::read(d.join("input.json")).unwrap()).unwrap();
    let expected = model.infer(&input).unwrap();
    let (result, costs) = laya_candle::profile::capture(counter, || model.infer(&input));
    assert_eq!(result.unwrap(), expected);
    assert!(costs.iter().any(|c| c.name == "int8.matmul"));
    assert!(costs.iter().any(|c| c.name == "attention.qk"));
    assert!(costs.iter().all(|c| c.instructions > 0));
    let (_, costs) = laya_candle::profile::capture(counter, || ());
    assert!(costs.is_empty());
}

#[test]
fn tiled_int8_matches_integer_reference_including_all_tails() {
    use candle_core::{Device, Tensor};
    for cols in [1, 15, 16, 17, 31, 32, 33, 1024, 2624, 16384] {
        for (tokens, rows) in [(3, 3), (4, 4), (5, 7), (8, 9)] {
            let w: Vec<i8> = (0..rows*cols).map(|i| ((i*37+11)%256) as u8 as i8).collect();
            let x: Vec<f32> = (0..tokens*cols).map(|i| (((i*19)%255) as i32-127) as f32).collect();
            let mut bytes: Vec<u8> = w.iter().map(|v| *v as u8).collect();
            for _ in 0..rows { bytes.extend(0.25f32.to_le_bytes()); }
            let matrix = laya_candle::int8::Int8Matrix::from_bytes(&bytes, rows, cols).unwrap();
            let tensor = Tensor::from_vec(x.clone(), (tokens, cols), &Device::Cpu).unwrap();
            let actual = matrix.forward(&tensor).unwrap().to_vec2::<f32>().unwrap();
            for t in 0..tokens {
                let a=&x[t*cols..(t+1)*cols];
                let scale=a.iter().fold(0f32,|a,b|a.max(b.abs()))/127.;
                for r in 0..rows {
                    let sum:i64=a.iter().zip(&w[r*cols..(r+1)*cols]).map(|(a,b)| (a/scale).round().clamp(-127.,127.) as i64 * *b as i64).sum();
                    assert_eq!(actual[t][r],sum as f32*scale*0.25,"T={tokens} O={rows} K={cols}");
                }
            }
        }
    }
}

#[test]
fn rejects_legacy_and_relabelled_f32_packs() {
    use laya_candle::pack::{Builder, Manifest};
    let raw = std::fs::read(dir("tiny-prenorm").join("manifest.json")).unwrap();
    assert!(Builder::new(&raw).is_err());
    let mut manifest: Manifest = serde_json::from_slice(&raw).unwrap();
    manifest.format = "ic-laya-int8-pack-v1".into();
    assert!(manifest.validate().is_err());
    assert!(Builder::new(&serde_json::to_vec(&manifest).unwrap()).is_err());
}

#[test]
fn direct_constructor_rejects_f32_matrices() {
    use candle_core::{Device, DType, Tensor};
    use laya_candle::{LayaModel, Weight, pack::Manifest};
    let raw = std::fs::read(dir("tiny-prenorm").join("manifest.json")).unwrap();
    let manifest: Manifest = serde_json::from_slice(&raw).unwrap();
    let weights = manifest.tensors.iter().map(|entry| {
        let tensor = Tensor::zeros(entry.shape.as_slice(), DType::F32, &Device::Cpu).unwrap();
        (entry.name.clone(), Weight::F32(tensor))
    }).collect();
    assert!(matches!(LayaModel::from_weights(manifest.config, [0;32], BackendKind::SyntheticFixture, weights), Err(Error::Invalid(_))));
}
