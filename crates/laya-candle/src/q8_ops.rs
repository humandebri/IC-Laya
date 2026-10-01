//! Exact elementwise SIMD preparation for the experimental INT8 path.

pub(crate) fn dequantize(src: &[i8], scale: f32, out: &mut [f32]) {
    assert_eq!(src.len(), out.len());
    #[cfg(target_arch = "wasm32")]
    return wasm::dequantize(src, scale, out);
    #[cfg(not(target_arch = "wasm32"))]
    for (o, &v) in out.iter_mut().zip(src) {
        *o = v as f32 * scale;
    }
}
pub(crate) fn scale_i32(src: &[i32], scale: f32, out: &mut [f32]) {
    assert_eq!(src.len(), out.len());
    #[cfg(target_arch = "wasm32")]
    return wasm::scale_i32(src, scale, out);
    #[cfg(not(target_arch = "wasm32"))]
    for (o, &v) in out.iter_mut().zip(src) {
        *o = v as f32 * scale;
    }
}
pub(crate) fn rotary(src: &[i8], scale: f32, rotations: &[[f32; 2]], out: &mut [f32]) {
    assert_eq!(src.len(), out.len());
    assert_eq!(src.len(), rotations.len() * 2);
    #[cfg(target_arch = "wasm32")]
    return wasm::rotary(src, scale, rotations, out);
    #[cfg(not(target_arch = "wasm32"))]
    scalar_rotary(src, scale, rotations, out, 0);
}
fn scalar_rotary(src: &[i8], scale: f32, rotations: &[[f32; 2]], out: &mut [f32], start: usize) {
    let half = rotations.len();
    for i in start..half {
        let [sin, cos] = rotations[i];
        let a = src[i] as f32 * scale;
        let b = src[i + half] as f32 * scale;
        out[i] = a * cos - b * sin;
        out[i + half] = b * cos + a * sin;
    }
}
pub(crate) fn statistics(src: &[i8]) -> (i64, i64) {
    assert!(!src.is_empty() && src.len() <= 16384);
    #[cfg(target_arch = "wasm32")]
    return wasm::statistics(src);
    #[cfg(not(target_arch = "wasm32"))]
    src.iter().fold((0, 0), |(sum, squares), &v| {
        let v = v as i64;
        (sum + v, squares + v * v)
    })
}
pub(crate) fn normalize_affine(
    src: &[i8],
    mean: f64,
    inv: f64,
    w: &[f32],
    b: Option<&[f32]>,
    out: &mut [f32],
) {
    assert_eq!(src.len(), out.len());
    assert_eq!(w.len(), out.len());
    assert!(b.is_none_or(|b| b.len() == out.len()));
    #[cfg(target_arch = "wasm32")]
    return wasm::normalize_affine(src, mean, inv, w, b, out);
    #[cfg(not(target_arch = "wasm32"))]
    for c in 0..src.len() {
        out[c] = (((src[c] as f64 - mean) * inv) as f32) * w[c] + b.map_or(0., |b| b[c]);
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
mod wasm {
    use std::arch::wasm32::*;
    pub(super) fn dequantize(src: &[i8], scale: f32, out: &mut [f32]) {
        #[target_feature(enable = "simd128")]
        unsafe fn run(src: &[i8], scale: f32, out: &mut [f32]) {
            let mut i = 0;
            let s = f32x4_splat(scale);
            while i + 16 <= src.len() {
                let v = unsafe { v128_load(src.as_ptr().add(i).cast()) };
                let lo = i16x8_extend_low_i8x16(v);
                let hi = i16x8_extend_high_i8x16(v);
                let values = [
                    i32x4_extend_low_i16x8(lo),
                    i32x4_extend_high_i16x8(lo),
                    i32x4_extend_low_i16x8(hi),
                    i32x4_extend_high_i16x8(hi),
                ];
                for j in 0..4 {
                    unsafe {
                        v128_store(
                            out.as_mut_ptr().add(i + 4 * j).cast(),
                            f32x4_mul(f32x4_convert_i32x4(values[j]), s),
                        )
                    };
                }
                i += 16;
            }
            while i < src.len() {
                out[i] = src[i] as f32 * scale;
                i += 1;
            }
        }
        // Equal lengths checked by caller; complete 16-byte/64-byte chunks only.
        unsafe { run(src, scale, out) }
    }
    pub(super) fn scale_i32(src: &[i32], scale: f32, out: &mut [f32]) {
        #[target_feature(enable = "simd128")]
        unsafe fn run(src: &[i32], scale: f32, out: &mut [f32]) {
            let mut i = 0;
            let s = f32x4_splat(scale);
            while i + 4 <= src.len() {
                let v = unsafe { v128_load(src.as_ptr().add(i).cast()) };
                unsafe {
                    v128_store(
                        out.as_mut_ptr().add(i).cast(),
                        f32x4_mul(f32x4_convert_i32x4(v), s),
                    )
                };
                i += 4;
            }
            while i < src.len() {
                out[i] = src[i] as f32 * scale;
                i += 1;
            }
        }
        // Equal slice lengths checked; each vector spans four initialized values.
        unsafe { run(src, scale, out) }
    }
    pub(super) fn rotary(src: &[i8], scale: f32, rotations: &[[f32; 2]], out: &mut [f32]) {
        #[target_feature(enable = "simd128")]
        unsafe fn run(src: &[i8], scale: f32, rotations: &[[f32; 2]], out: &mut [f32]) {
            let half = rotations.len();
            let mut i = 0;
            let s = f32x4_splat(scale);
            while i + 4 <= half {
                // Checked half length bounds both four-byte signed input loads.
                let a = unsafe { v128_load32_zero(src.as_ptr().add(i).cast()) };
                let b = unsafe { v128_load32_zero(src.as_ptr().add(i + half).cast()) };
                let a = f32x4_mul(
                    f32x4_convert_i32x4(i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(a))),
                    s,
                );
                let b = f32x4_mul(
                    f32x4_convert_i32x4(i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(b))),
                    s,
                );
                // Four [sin,cos] arrays are eight contiguous floats.
                let r0 = unsafe { v128_load(rotations.as_ptr().add(i).cast()) };
                let r1 = unsafe { v128_load(rotations.as_ptr().add(i + 2).cast()) };
                let sin = i32x4_shuffle::<0, 2, 4, 6>(r0, r1);
                let cos = i32x4_shuffle::<1, 3, 5, 7>(r0, r1);
                // Preserve original multiply then add/subtract order, with no FMA.
                unsafe {
                    v128_store(
                        out.as_mut_ptr().add(i).cast(),
                        f32x4_sub(f32x4_mul(a, cos), f32x4_mul(b, sin)),
                    )
                };
                unsafe {
                    v128_store(
                        out.as_mut_ptr().add(i + half).cast(),
                        f32x4_add(f32x4_mul(b, cos), f32x4_mul(a, sin)),
                    )
                };
                i += 4;
            }
            super::scalar_rotary(src, scale, rotations, out, i);
        }
        // Caller checks input/output and two-value rotation-pair lengths.
        unsafe { run(src, scale, rotations, out) }
    }
    pub(super) fn statistics(src: &[i8]) -> (i64, i64) {
        #[target_feature(enable = "simd128")]
        unsafe fn run(src: &[i8]) -> (i64, i64) {
            let mut i = 0;
            let mut sum = i32x4_splat(0);
            let mut squares = i32x4_splat(0);
            while i + 16 <= src.len() {
                let v = unsafe { v128_load(src.as_ptr().add(i).cast()) };
                let lo = i16x8_extend_low_i8x16(v);
                let hi = i16x8_extend_high_i8x16(v);
                sum = i32x4_add(
                    sum,
                    i32x4_add(
                        i32x4_extadd_pairwise_i16x8(lo),
                        i32x4_extadd_pairwise_i16x8(hi),
                    ),
                );
                squares = i32x4_add(
                    squares,
                    i32x4_add(i32x4_dot_i16x8(lo, lo), i32x4_dot_i16x8(hi, hi)),
                );
                i += 16;
            }
            let mut sum = i32x4_extract_lane::<0>(sum) as i64
                + i32x4_extract_lane::<1>(sum) as i64
                + i32x4_extract_lane::<2>(sum) as i64
                + i32x4_extract_lane::<3>(sum) as i64;
            let mut squares = i32x4_extract_lane::<0>(squares) as i64
                + i32x4_extract_lane::<1>(squares) as i64
                + i32x4_extract_lane::<2>(squares) as i64
                + i32x4_extract_lane::<3>(squares) as i64;
            while i < src.len() {
                let v = src[i] as i64;
                sum += v;
                squares += v * v;
                i += 1;
            }
            (sum, squares)
        }
        // Caller bounds K<=16384: each i32 square lane <= 2^26, sums <= 2^19.
        // Complete 16-byte chunks only; tails retain scalar exact integer sums.
        unsafe { run(src) }
    }
    pub(super) fn normalize_affine(
        src: &[i8],
        mean: f64,
        inv: f64,
        w: &[f32],
        b: Option<&[f32]>,
        out: &mut [f32],
    ) {
        #[target_feature(enable = "simd128")]
        unsafe fn run(
            src: &[i8],
            mean: f64,
            inv: f64,
            w: &[f32],
            b: Option<&[f32]>,
            out: &mut [f32],
        ) {
            let mut i = 0;
            let m = f64x2_splat(mean);
            let scale = f64x2_splat(inv);
            while i + 4 <= src.len() {
                let v = unsafe { v128_load32_zero(src.as_ptr().add(i).cast()) };
                let ints = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(v));
                let lo = f64x2_mul(f64x2_sub(f64x2_convert_low_i32x4(ints), m), scale);
                let hi = f64x2_mul(
                    f64x2_sub(
                        f64x2_convert_low_i32x4(i32x4_shuffle::<2, 3, 0, 1>(ints, ints)),
                        m,
                    ),
                    scale,
                );
                let normalized = i32x4_shuffle::<0, 1, 4, 5>(
                    f32x4_demote_f64x2_zero(lo),
                    f32x4_demote_f64x2_zero(hi),
                );
                let weight = unsafe { v128_load(w.as_ptr().add(i).cast()) };
                let bias = match b {
                    Some(b) => unsafe { v128_load(b.as_ptr().add(i).cast()) },
                    None => f32x4_splat(0.),
                };
                unsafe {
                    v128_store(
                        out.as_mut_ptr().add(i).cast(),
                        f32x4_add(f32x4_mul(normalized, weight), bias),
                    )
                };
                i += 4;
            }
            while i < src.len() {
                out[i] = (((src[i] as f64 - mean) * inv) as f32) * w[i] + b.map_or(0., |b| b[i]);
                i += 1;
            }
        }
        // All lengths checked; four bytes in input and four f32 output/weights.
        // F64 operations, F32 cast and affine order match the scalar path.
        unsafe { run(src, mean, inv, w, b, out) }
    }
}
