//! Runs the actual target-specific Int8Matrix kernel against independent i64 sums.
use candle_core::{Device, Tensor};
use laya_candle::int8::Int8Matrix;

#[no_mangle]
pub extern "C" fn check() -> u32 {
    let mut cases = 0;
    for cols in [1, 15, 16, 17, 31, 32, 33, 1024, 2624, 16384] {
        for (tokens, rows) in [
            (1, 1),
            (2, 3),
            (3, 7),
            (4, 8),
            (5, 9),
            (7, 15),
            (8, 16),
            (9, 17),
            (15, 16),
            (16, 17),
            (17, 16),
            (31, 8),
            (32, 16),
            (33, 17),
            (63, 16),
            (64, 16),
            (65, 16),
            (85, 16),
            (86, 16),
            (87, 16),
            (88, 16),
            (127, 16),
            (128, 16),
        ] {
            for pattern in 0..4 {
                let weights: Vec<i8> = (0..rows * cols)
                    .map(|i| match pattern {
                        0 => ((i * 37 + 11) % 256) as u8 as i8,
                        1 => -128,
                        2 => 127,
                        _ => 0,
                    })
                    .collect();
                let input: Vec<f32> = (0..tokens * cols)
                    .map(|i| match pattern {
                        0 => ((i * 19 % 255) as i32 - 127) as f32,
                        1 => -127.,
                        2 => 127.,
                        _ => 0.,
                    })
                    .collect();
                let mut bytes: Vec<u8> = weights.iter().map(|v| *v as u8).collect();
                for _ in 0..rows {
                    bytes.extend(0.25f32.to_le_bytes());
                }
                let matrix = Int8Matrix::from_bytes(&bytes, rows, cols).unwrap();
                let tensor = Tensor::from_vec(input.clone(), (tokens, cols), &Device::Cpu).unwrap();
                let actual = matrix.forward(&tensor).unwrap().to_vec2::<f32>().unwrap();
                for t in 0..tokens {
                    let row = &input[t * cols..(t + 1) * cols];
                    let peak = row.iter().fold(0f32, |a, b| a.max(b.abs()));
                    let scale = if peak == 0. { 1. } else { peak / 127. };
                    for r in 0..rows {
                        let sum: i64 = row
                            .iter()
                            .zip(&weights[r * cols..(r + 1) * cols])
                            .map(|(a, b)| (a / scale).round().clamp(-127., 127.) as i64 * *b as i64)
                            .sum();
                        assert_eq!(
                            actual[t][r],
                            sum as f32 * scale * 0.25,
                            "T={tokens} O={rows} K={cols} pattern={pattern}"
                        );
                    }
                }
                cases += 1;
            }
        }
    }
    cases
}

/// Identity weights expose each quantized activation, including half-way boundaries.
#[no_mangle]
pub extern "C" fn check_quantization() -> u32 {
    let mut cases = 0;
    let edge = [
        0.,
        -0.,
        0.5,
        -0.5,
        f32::from_bits(0x3effffff),
        -f32::from_bits(0x3effffff),
        f32::from_bits(0x3f000001),
        -f32::from_bits(0x3f000001),
        1.5,
        -1.5,
        126.5,
        -126.5,
        127.,
        -127.,
    ];
    for width in [4, 5, 15, 16, 17, 32] {
        let mut bytes = vec![0u8; width * width];
        for i in 0..width {
            bytes[i * width + i] = 1;
        }
        for _ in 0..width {
            bytes.extend(1f32.to_le_bytes());
        }
        let w = Int8Matrix::from_bytes(&bytes, width, width).unwrap();
        for pattern in 0..18 {
            let mut row: Vec<f32> = (0..width)
                .map(|i| match pattern {
                    14 => {
                        if i % 2 == 0 {
                            f32::from_bits(1)
                        } else {
                            -f32::from_bits(1)
                        }
                    }
                    15 => {
                        if i % 2 == 0 {
                            f32::MIN_POSITIVE
                        } else {
                            -f32::MIN_POSITIVE
                        }
                    }
                    16 => {
                        if i % 2 == 0 {
                            f32::MAX
                        } else {
                            -f32::MAX
                        }
                    }
                    17 => 0.,
                    _ => edge[(i + pattern) % edge.len()],
                })
                .collect();
            if pattern < 14 {
                row[width - 1] = 127.;
            }
            let peak = row.iter().fold(0f32, |a, b| a.max(b.abs()));
            let scale = if peak == 0. {
                1.
            } else {
                (peak / 127.).max(f32::MIN_POSITIVE)
            };
            let x = Tensor::from_vec(row.clone(), (1, width), &Device::Cpu).unwrap();
            let actual = w
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (a, b) in actual.iter().zip(&row) {
                let q = (b / scale).round().clamp(-127., 127.) as i8;
                assert_eq!(a.to_bits(), (q as f32 * scale).to_bits());
            }
            cases += 1;
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, width - 1] {
                let mut row = vec![0f32; width];
                row[index] = value;
                assert!(w
                    .forward(&Tensor::from_vec(row, (1, width), &Device::Cpu).unwrap())
                    .is_err());
                cases += 1;
            }
        }
    }
    cases
}

/// Exercise non-power-of-two activation and row scales at the Wasm F32 writeback.
#[no_mangle]
pub extern "C" fn check_writeback() -> u32 {
    let mut cases = 0;
    for tokens in [1, 3, 16, 17, 87, 128] {
        for rows in [1, 4, 8, 16, 17, 32] {
            for cols in [1, 15, 17, 32, 1024, 2624] {
                let weights: Vec<i8> = (0..rows * cols)
                    .map(|i| ((i * 73 + 59) % 256) as u8 as i8)
                    .collect();
                let row_scales: Vec<f32> = (0..rows)
                    .map(|r| 0.00073f32 + ((r * 19) % 33) as f32 * 0.011)
                    .collect();
                let mut bytes: Vec<u8> = weights.iter().map(|v| *v as u8).collect();
                for scale in &row_scales {
                    bytes.extend(scale.to_le_bytes());
                }
                let input: Vec<f32> = (0..tokens * cols)
                    .map(|i| ((i * 89 % 245) as i32 - 122) as f32 * 0.019 + 0.0037)
                    .collect();
                let matrix = Int8Matrix::from_bytes(&bytes, rows, cols).unwrap();
                let tensor = Tensor::from_vec(input.clone(), (tokens, cols), &Device::Cpu).unwrap();
                let actual = matrix.forward(&tensor).unwrap().to_vec2::<f32>().unwrap();
                for t in 0..tokens {
                    let row = &input[t * cols..(t + 1) * cols];
                    let peak = row.iter().fold(0f32, |a, b| a.max(b.abs()));
                    let scale = (peak / 127.).max(f32::MIN_POSITIVE);
                    for r in 0..rows {
                        let sum: i64 = row
                            .iter()
                            .zip(&weights[r * cols..(r + 1) * cols])
                            .map(|(a, b)| (a / scale).round().clamp(-127., 127.) as i64 * *b as i64)
                            .sum();
                        let expected = sum as f32 * scale * row_scales[r];
                        assert_eq!(
                            actual[t][r].to_bits(),
                            expected.to_bits(),
                            "T={tokens} O={rows} K={cols} row={r}"
                        );
                    }
                }
                cases += 1;
            }
        }
    }
    cases
}

// Compile the exact shared SIMD source under the Wasm target, including tails.
#[path = "../src/q8_ops.rs"]
mod q8_ops;
#[no_mangle]
pub extern "C" fn check_q8_ops() -> u32 {
    let mut cases = 0;
    for width in [2, 4, 6, 8, 10, 16, 32, 64, 128] {
        for scale in [0., 0.0031, 0.25, 1e-20] {
            for pattern in 0..4 {
                let src: Vec<i8> = (0..width)
                    .map(|i| match pattern {
                        0 => -128,
                        1 => 127,
                        2 => ((i * 37 + 11) % 256) as u8 as i8,
                        _ => 0,
                    })
                    .collect();
                let dots: Vec<i32> = (0..width)
                    .map(|i| {
                        if i % 2 == 0 {
                            i32::MIN + i as i32
                        } else {
                            i32::MAX - i as i32
                        }
                    })
                    .collect();
                let rotations: Vec<[f32; 2]> = (0..width / 2)
                    .map(|i| {
                        let (sin, cos) = (i as f64 * 0.317).sin_cos();
                        [sin as f32, cos as f32]
                    })
                    .collect();
                let mut out = vec![0.; width];
                let (sum, squares) = q8_ops::statistics(&src);
                let expected_sum: i64 = src.iter().map(|&v| v as i64).sum();
                let expected_squares: i64 = src.iter().map(|&v| (v as i64) * (v as i64)).sum();
                assert_eq!((sum, squares), (expected_sum, expected_squares));
                let mean = sum as f64 / width as f64;
                let inv = 0.31743;
                let weight: Vec<f32> = (0..width).map(|i| (i as f32 - 3.) * 0.019).collect();
                let bias: Vec<f32> = (0..width).map(|i| (i as f32 - 5.) * 0.0011).collect();
                for b in [None, Some(bias.as_slice())] {
                    q8_ops::normalize_affine(&src, mean, inv, &weight, b, &mut out);
                    for i in 0..width {
                        let value = (((src[i] as f64 - mean) * inv) as f32) * weight[i]
                            + b.map_or(0., |b| b[i]);
                        assert_eq!(out[i].to_bits(), value.to_bits());
                    }
                }
                q8_ops::dequantize(&src, scale, &mut out);
                for i in 0..width {
                    assert_eq!(out[i].to_bits(), (src[i] as f32 * scale).to_bits());
                }
                q8_ops::scale_i32(&dots, scale, &mut out);
                for i in 0..width {
                    assert_eq!(out[i].to_bits(), (dots[i] as f32 * scale).to_bits());
                }
                q8_ops::rotary(&src, scale, &rotations, &mut out);
                for i in 0..width {
                    let half = width / 2;
                    let [sin, cos] = rotations[i % half];
                    let a = src[i] as f32 * scale;
                    let b = src[if i < half { i + half } else { i - half }] as f32 * scale;
                    let value = if i < half {
                        a * cos - b * sin
                    } else {
                        a * cos + b * sin
                    };
                    assert_eq!(out[i].to_bits(), value.to_bits());
                }
                cases += 1;
            }
        }
    }
    cases
}

#[no_mangle]
pub extern "C" fn check_q8_statistics() -> u32 {
    let mut count = 0;
    for width in [1, 15, 16, 17, 1024, 16384] {
        for pattern in 0..4 {
            let src: Vec<i8> = (0..width)
                .map(|i| match pattern {
                    0 => -128,
                    1 => 127,
                    2 => ((i * 19 + 3) % 256) as u8 as i8,
                    _ => 0,
                })
                .collect();
            let a: i64 = src.iter().map(|&v| v as i64).sum();
            let b: i64 = src.iter().map(|&v| (v as i64) * (v as i64)).sum();
            assert_eq!(q8_ops::statistics(&src), (a, b));
            count += 1;
        }
    }
    count
}
