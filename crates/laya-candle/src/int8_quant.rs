//! Exact half-away-from-zero dynamic activation quantization.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn row(input: &[f32], out: &mut [i8]) -> candle_core::Result<f32> {
    assert_eq!(input.len(), out.len());
    if input.iter().any(|v| !v.is_finite()) {
        candle_core::bail!("nonfinite int8 activation")
    }
    let peak = input.iter().fold(0f32, |a, b| a.max(b.abs()));
    let scale = if peak == 0. {
        1.
    } else {
        (peak / 127.).max(f32::MIN_POSITIVE)
    };
    for (q, v) in out.iter_mut().zip(input) {
        *q = (v / scale).round().clamp(-127., 127.) as i8;
    }
    Ok(scale)
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
pub(super) fn row(input: &[f32], out: &mut [i8]) -> candle_core::Result<f32> {
    assert_eq!(input.len(), out.len());
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn kernel(input: &[f32], out: &mut [i8]) -> candle_core::Result<f32> {
        let end = input.len() / 4 * 4;
        let mut peak = f32x4_splat(0.);
        for i in (0..end).step_by(4) {
            // Four initialized f32 values, within the input slice.
            let values = f32x4_abs(unsafe { v128_load(input.as_ptr().add(i).cast()) });
            if i32x4_bitmask(f32x4_le(values, f32x4_splat(f32::MAX))) != 15 {
                candle_core::bail!("nonfinite int8 activation")
            }
            peak = f32x4_max(peak, values);
        }
        let mut peak = f32x4_extract_lane::<0>(peak)
            .max(f32x4_extract_lane::<1>(peak))
            .max(f32x4_extract_lane::<2>(peak))
            .max(f32x4_extract_lane::<3>(peak));
        for value in &input[end..] {
            if !value.is_finite() {
                candle_core::bail!("nonfinite int8 activation")
            }
            peak = peak.max(value.abs());
        }
        let scale = if peak == 0. {
            1.
        } else {
            (peak / 127.).max(f32::MIN_POSITIVE)
        };
        for i in (0..end).step_by(4) {
            let x = f32x4_div(
                unsafe { v128_load(input.as_ptr().add(i).cast()) },
                f32x4_splat(scale),
            );
            let mag = f32x4_abs(x);
            let whole = f32x4_trunc(mag);
            // Do not use nearest (ties-even), reciprocal multiplication, or abs(x)+0.5.
            // The latter can round a value just below a half to the next integer.
            let rounded = f32x4_add(
                whole,
                v128_and(
                    f32x4_ge(f32x4_sub(mag, whole), f32x4_splat(0.5)),
                    f32x4_splat(1.),
                ),
            );
            let signed = v128_or(rounded, v128_and(x, i32x4_splat(i32::MIN)));
            let q = i32x4_trunc_sat_f32x4(f32x4_max(
                f32x4_splat(-127.),
                f32x4_min(signed, f32x4_splat(127.)),
            ));
            let packed=i8x16_narrow_i16x8(i16x8_narrow_i32x4(q,i32x4_splat(0)),i16x8_splat(0));
            // Four output bytes, equal lengths checked; unaligned stores allowed.
            unsafe {std::ptr::write_unaligned(out.as_mut_ptr().add(i).cast::<i32>(),i32x4_extract_lane::<0>(packed))};
        }
        for (q, v) in out[end..].iter_mut().zip(&input[end..]) {
            *q = (v / scale).round().clamp(-127., 127.) as i8;
        }
        Ok(scale)
    }
    // Equal lengths checked; complete vector chunks bounded by end. IC supports simd128.
    unsafe { kernel(input, out) }
}
