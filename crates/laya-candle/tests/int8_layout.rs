use candle_core::{Device, Tensor};
use laya_candle::int8::Int8Matrix;

#[test]
fn input_views_match_owned_contiguous_inputs() {
    let mut bytes = vec![1, 255, 2, 254, 3, 253, 4, 252];
    bytes.extend(0.25f32.to_le_bytes());
    bytes.extend(0.5f32.to_le_bytes());
    let matrix = Int8Matrix::from_bytes(&bytes, 2, 4).unwrap();
    let base = Tensor::from_vec(
        (0..24).map(|i| (i as f32 - 12.) / 7.).collect(),
        (6, 4),
        &Device::Cpu,
    )
    .unwrap();
    let transposed = base.transpose(0, 1).unwrap().narrow(1, 1, 4).unwrap();
    for view in [base.narrow(0, 1, 3).unwrap(), transposed] {
        let rows = view.to_vec2::<f32>().unwrap();
        let owned = Tensor::from_vec(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            view.shape(),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            matrix.forward(&view).unwrap().to_vec2::<f32>().unwrap(),
            matrix.forward(&owned).unwrap().to_vec2::<f32>().unwrap()
        );
    }
}

#[test]
fn borrowed_input_still_rejects_nonfinite_values() {
    let mut bytes = vec![1; 4];
    bytes.extend(1f32.to_le_bytes());
    let matrix = Int8Matrix::from_bytes(&bytes, 1, 4).unwrap();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = Tensor::from_vec(vec![0., value, 1., -1.], (1, 4), &Device::Cpu).unwrap();
        assert!(matrix.forward(&input).is_err());
    }
}
