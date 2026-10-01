//! Bounded owner-diagnostic epilogue experiments. Fixed scales are not enabled
//! in model inference. Inputs contain real INT32 sums and explicit calibration.
use candle_core::Result;
use ic_laya_core::{hash, Digest};

pub const MAX_BYTES: usize = 16 + 128 * 5248 * 4 + (128 + 5248 + 48 + 5248) * 4;
const Q: f32 = 2147483648.;
const HALF: i64 = 1 << 30;
const MAX_SUM: i64 = 1 << 28;
const BIAS_LIMIT: i64 = i64::MAX - (1i64 << 59) - HALF;

pub struct Input {
    pub tokens: usize,
    pub outputs: usize,
    pub groups: usize,
    pub sums: Vec<i32>,
    pub sx: Vec<f32>,
    pub sw: Vec<f32>,
    pub sy: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}
pub struct Variant {
    pub name: String,
    pub instructions: u64,
    pub checksum: Digest,
    pub clamped: u64,
    pub rmse: f64,
    pub max_abs_error: f32,
    pub different_from_fixed_float: u64,
}
pub struct Report {
    pub prepare_instructions: u64,
    pub variants: Vec<Variant>,
}

impl Input {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || bytes.len() > MAX_BYTES {
            candle_core::bail!("epilogue byte length")
        }
        let read = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let (tokens, outputs, groups, bias) = (read(0), read(4), read(8), read(12));
        if !(1..=128).contains(&tokens)
            || ![(1024, 1), (3072, 48), (5248, 2)].contains(&(outputs, groups))
            || bias > 1
        {
            candle_core::bail!("epilogue shape")
        }
        let count = tokens * outputs;
        if bytes.len() != 16 + 4 * (count + tokens + outputs + groups + bias * outputs) {
            candle_core::bail!("epilogue payload")
        }
        let sums = bytes[16..16 + count * 4]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut offset = 16 + count * 4;
        let mut floats = |n| {
            let v = bytes[offset..offset + 4 * n]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            offset += 4 * n;
            v
        };
        let sx = floats(tokens);
        let sw = floats(outputs);
        let sy = floats(groups);
        let bias = if bias == 1 {
            Some(floats(outputs))
        } else {
            None
        };
        let input = Self {
            tokens,
            outputs,
            groups,
            sums,
            sx,
            sw,
            sy,
            bias,
        };
        input.validate()?;
        Ok(input)
    }
    fn validate(&self) -> Result<()> {
        if self.tokens == 0
            || self.outputs == 0
            || self.groups == 0
            || self.outputs % self.groups != 0
            || self.sums.len() != self.tokens * self.outputs
            || self.sx.len() != self.tokens
            || self.sw.len() != self.outputs
            || self.sy.len() != self.groups
            || self.bias.as_ref().is_some_and(|b| b.len() != self.outputs)
        {
            candle_core::bail!("epilogue lengths")
        }
        if self.sums.iter().any(|&v| (v as i64).abs() > MAX_SUM)
            || self
                .sx
                .iter()
                .chain(&self.sw)
                .chain(&self.sy)
                .any(|&v| !v.is_finite() || v <= 0.)
            || self
                .bias
                .as_ref()
                .is_some_and(|b| b.iter().any(|v| !v.is_finite()))
        {
            candle_core::bail!("epilogue numeric input")
        }
        Ok(())
    }
    fn physical(&self, t: usize, c: usize) -> f32 {
        self.sums[t * self.outputs + c] as f32 * self.sx[t] * self.sw[c]
            + self.bias.as_ref().map_or(0., |b| b[c])
    }
}

struct Prepared {
    coeff: Vec<f32>,
    bias: Vec<i64>,
}
fn prepare(input: &Input) -> Result<Prepared> {
    let width = input.outputs / input.groups;
    let peak = input.sx.iter().copied().fold(0f32, f32::max);
    let mut coeff = Vec::with_capacity(input.outputs);
    let mut bias = Vec::with_capacity(input.outputs);
    for c in 0..input.outputs {
        let scale = input.sy[c / width];
        let ratio = input.sw[c] / scale;
        if !ratio.is_finite() || (peak * ratio) * Q >= Q {
            candle_core::bail!("Q31 coefficient range")
        }
        let b = input.bias.as_ref().map_or(0., |b| b[c]) / scale;
        let fixed = (b as f64 * Q as f64).round();
        if !fixed.is_finite() || fixed.abs() >= BIAS_LIMIT as f64 {
            candle_core::bail!("Q31 bias range")
        }
        coeff.push(ratio);
        bias.push(fixed as i64);
    }
    Ok(Prepared { coeff, bias })
}
fn shifted(n: i64) -> (i8, bool) {
    let mag = (n.unsigned_abs() + (HALF as u64)) >> 31;
    let clipped = mag > 127;
    let q = mag.min(127) as i8;
    (if n < 0 { -q } else { q }, clipped)
}
fn fixed_integer(input: &Input, p: &Prepared) -> (Vec<i8>, u64) {
    let mut out = vec![0i8; input.sums.len()];
    let mut clipped = 0;
    for t in 0..input.tokens {
        clipped += integer_row(
            &input.sums[t * input.outputs..(t + 1) * input.outputs],
            input.sx[t],
            &p.coeff,
            &p.bias,
            &mut out[t * input.outputs..(t + 1) * input.outputs],
        );
    }
    (out, clipped)
}
#[cfg(not(target_arch = "wasm32"))]
fn integer_row(sums: &[i32], sx: f32, coeff: &[f32], bias: &[i64], out: &mut [i8]) -> u64 {
    let mut clipped = 0;
    for c in 0..out.len() {
        let m = ((sx * coeff[c]) * Q).round() as i32;
        let (q, clip) = shifted(sums[c] as i64 * m as i64 + bias[c]);
        out[c] = q;
        clipped += clip as u64;
    }
    clipped
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
fn integer_row(sums: &[i32], sx: f32, coeff: &[f32], bias: &[i64], out: &mut [i8]) -> u64 {
    use std::arch::wasm32::*;
    assert_eq!(sums.len(), out.len());
    assert_eq!(coeff.len(), out.len());
    assert_eq!(bias.len(), out.len());
    #[target_feature(enable = "simd128")]
    unsafe fn kernel(sums: &[i32], sx: f32, coeff: &[f32], bias: &[i64], out: &mut [i8]) -> u64 {
        fn rounded(n: v128) -> (v128, u64) {
            let negative = i64x2_lt(n, i64x2_splat(0));
            let mag = i64x2_shr(i64x2_add(i64x2_abs(n), i64x2_splat(HALF)), 31);
            let clipped = i64x2_gt(mag, i64x2_splat(127));
            let q = v128_bitselect(i64x2_splat(127), mag, clipped);
            (
                v128_bitselect(i64x2_neg(q), q, negative),
                i64x2_bitmask(clipped).count_ones() as u64,
            )
        }
        let mut c = 0;
        let mut clips = 0;
        let sx = f32x4_splat(sx);
        while c + 4 <= out.len() {
            // Equal lengths checked; each load/store covers a complete chunk.
            let dots = unsafe { v128_load(sums.as_ptr().add(c).cast()) };
            let ratios = unsafe { v128_load(coeff.as_ptr().add(c).cast()) };
            let f = f32x4_mul(f32x4_mul(sx, ratios), f32x4_splat(Q));
            let whole = f32x4_trunc(f);
            let m = i32x4_trunc_sat_f32x4(f32x4_add(
                whole,
                v128_and(
                    f32x4_ge(f32x4_sub(f, whole), f32x4_splat(0.5)),
                    f32x4_splat(1.),
                ),
            ));
            let lo = unsafe { v128_load(bias.as_ptr().add(c).cast()) };
            let hi = unsafe { v128_load(bias.as_ptr().add(c + 2).cast()) };
            let (lo, a) = rounded(i64x2_add(i64x2_extmul_low_i32x4(dots, m), lo));
            let (hi, b) = rounded(i64x2_add(i64x2_extmul_high_i32x4(dots, m), hi));
            clips += a + b;
            let lanes = i32x4_shuffle::<0, 2, 4, 6>(lo, hi);
            let packed =
                i8x16_narrow_i16x8(i16x8_narrow_i32x4(lanes, i32x4_splat(0)), i16x8_splat(0));
            // Four initialized output bytes, still within out.len().
            unsafe {
                std::ptr::write_unaligned(
                    out.as_mut_ptr().add(c).cast::<i32>(),
                    i32x4_extract_lane::<0>(packed),
                )
            };
            c += 4;
        }
        while c < out.len() {
            let m = ((f32x4_extract_lane::<0>(sx) * coeff[c]) * Q).round() as i32;
            let (q, clip) = shifted(sums[c] as i64 * m as i64 + bias[c]);
            out[c] = q;
            clips += clip as u64;
            c += 1;
        }
        clips
    }
    // prepare() bounds products+bias away from i64 overflow, including rounding.
    unsafe { kernel(sums, sx, coeff, bias, out) }
}

fn fixed_float(input: &Input) -> (Vec<i8>, u64) {
    let mut out = vec![0; input.sums.len()];
    let width = input.outputs / input.groups;
    let mut clipped = 0;
    for t in 0..input.tokens {
        clipped += float_row(
            &input.sums[t * input.outputs..(t + 1) * input.outputs],
            input.sx[t],
            &input.sw,
            input.bias.as_deref(),
            &input.sy,
            width,
            &mut out[t * input.outputs..(t + 1) * input.outputs],
        );
    }
    (out, clipped)
}
#[cfg(not(target_arch = "wasm32"))]
fn float_row(
    sums: &[i32],
    sx: f32,
    sw: &[f32],
    bias: Option<&[f32]>,
    sy: &[f32],
    width: usize,
    out: &mut [i8],
) -> u64 {
    let mut clipped = 0;
    for c in 0..out.len() {
        let q = ((sums[c] as f32 * sx * sw[c] + bias.map_or(0., |b| b[c])) / sy[c / width]).round();
        clipped += (q.abs() > 127.) as u64;
        out[c] = q.clamp(-127., 127.) as i8;
    }
    clipped
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
fn float_row(
    sums: &[i32],
    sx: f32,
    sw: &[f32],
    bias: Option<&[f32]>,
    sy: &[f32],
    width: usize,
    out: &mut [i8],
) -> u64 {
    use std::arch::wasm32::*;
    assert_eq!(sums.len(), out.len());
    assert_eq!(sw.len(), out.len());
    if let Some(b) = bias {
        assert_eq!(b.len(), out.len());
    }
    assert_eq!(out.len(), width * sy.len());
    #[target_feature(enable = "simd128")]
    unsafe fn kernel(
        sums: &[i32],
        sx: f32,
        sw: &[f32],
        bias: Option<&[f32]>,
        sy: &[f32],
        width: usize,
        out: &mut [i8],
    ) -> u64 {
        let mut clipped = 0;
        for (g, &scale) in sy.iter().enumerate() {
            let end = (g + 1) * width;
            let mut c = g * width;
            while c + 4 <= end {
                // Complete chunks bounded by the output group and equal-length inputs.
                let dot = unsafe { v128_load(sums.as_ptr().add(c).cast()) };
                let w = unsafe { v128_load(sw.as_ptr().add(c).cast()) };
                let b = match bias {
                    Some(b) => unsafe { v128_load(b.as_ptr().add(c).cast()) },
                    None => f32x4_splat(0.),
                };
                let v = f32x4_div(
                    f32x4_add(
                        f32x4_mul(f32x4_mul(f32x4_convert_i32x4(dot), f32x4_splat(sx)), w),
                        b,
                    ),
                    f32x4_splat(scale),
                );
                let mag = f32x4_abs(v);
                let whole = f32x4_trunc(mag);
                let rounded = f32x4_add(
                    whole,
                    v128_and(
                        f32x4_ge(f32x4_sub(mag, whole), f32x4_splat(0.5)),
                        f32x4_splat(1.),
                    ),
                );
                clipped += i32x4_bitmask(f32x4_gt(rounded, f32x4_splat(127.))).count_ones() as u64;
                let signed = v128_or(rounded, v128_and(v, i32x4_splat(i32::MIN)));
                let q = i32x4_trunc_sat_f32x4(f32x4_max(
                    f32x4_splat(-127.),
                    f32x4_min(signed, f32x4_splat(127.)),
                ));
                let packed =
                    i8x16_narrow_i16x8(i16x8_narrow_i32x4(q, i32x4_splat(0)), i16x8_splat(0));
                unsafe {
                    std::ptr::write_unaligned(
                        out.as_mut_ptr().add(c).cast::<i32>(),
                        i32x4_extract_lane::<0>(packed),
                    )
                };
                c += 4;
            }
            while c < end {
                let q = ((sums[c] as f32 * sx * sw[c] + bias.map_or(0., |b| b[c])) / scale).round();
                clipped += (q.abs() > 127.) as u64;
                out[c] = q.clamp(-127., 127.) as i8;
                c += 1;
            }
        }
        clipped
    }
    unsafe { kernel(sums, sx, sw, bias, sy, width, out) }
}

// Matches the staged SIMD INT32-to-F32 writeback used by the old path.
#[allow(unsafe_code)]
fn staged_row(sums: &[i32], sx: f32, sw: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        assert_eq!(sums.len(), out.len());
        assert_eq!(sw.len(), out.len());
        assert!(bias.is_none_or(|b| b.len() == out.len()));
        #[target_feature(enable = "simd128")]
        unsafe fn run(sums: &[i32], sx: f32, sw: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
            let mut c = 0;
            while c + 4 <= out.len() {
                let dot = unsafe { v128_load(sums.as_ptr().add(c).cast()) };
                let weight = unsafe { v128_load(sw.as_ptr().add(c).cast()) };
                let mut v = f32x4_mul(f32x4_mul(f32x4_convert_i32x4(dot), f32x4_splat(sx)), weight);
                if let Some(b) = bias {
                    v = f32x4_add(v, unsafe { v128_load(b.as_ptr().add(c).cast()) });
                }
                unsafe { v128_store(out.as_mut_ptr().add(c).cast(), v) };
                c += 4;
            }
            while c < out.len() {
                out[c] = sums[c] as f32 * sx * sw[c] + bias.map_or(0., |b| b[c]);
                c += 1;
            }
        }
        // Complete chunks and equal slice lengths checked before SIMD accesses.
        unsafe { run(sums, sx, sw, bias, out) };
    }
    #[cfg(not(target_arch = "wasm32"))]
    for c in 0..out.len() {
        out[c] = sums[c] as f32 * sx * sw[c] + bias.map_or(0., |b| b[c]);
    }
}

pub fn benchmark(input: &Input, counter: fn() -> u64) -> Result<Report> {
    input.validate()?;
    let start = counter();
    let p = prepare(input)?;
    let prepare_instructions = counter().saturating_sub(start);
    let mut variants = Vec::new();
    let width = input.outputs / input.groups;
    let start = counter();
    let mut dynamic = vec![0i8; input.sums.len()];
    let mut dynamic_scales = vec![0f32; input.tokens * input.groups];
    let mut row = vec![0f32; input.outputs];
    for t in 0..input.tokens {
        staged_row(
            &input.sums[t * input.outputs..(t + 1) * input.outputs],
            input.sx[t],
            &input.sw,
            input.bias.as_deref(),
            &mut row,
        );
        for g in 0..input.groups {
            dynamic_scales[t * input.groups + g] = crate::int8::quantize_row(
                &row[g * width..(g + 1) * width],
                &mut dynamic[t * input.outputs + g * width..t * input.outputs + (g + 1) * width],
            )?;
        }
    }
    let dynamic_instructions = counter().saturating_sub(start);
    let start = counter();
    let (floating, float_clips) = fixed_float(input);
    let float_instructions = counter().saturating_sub(start);
    let start = counter();
    let (integer, int_clips) = fixed_integer(input, &p);
    let int_instructions = counter().saturating_sub(start);
    for (name, out, instructions, clamped) in [
        ("dynamic_staged", &dynamic, dynamic_instructions, 0),
        (
            "fixed_float_fused",
            &floating,
            float_instructions,
            float_clips,
        ),
        ("fixed_q31_fused", &integer, int_instructions, int_clips),
    ] {
        let mut squares = 0.;
        let mut max_error = 0f32;
        for t in 0..input.tokens {
            for c in 0..input.outputs {
                let scale = if name == "dynamic_staged" {
                    dynamic_scales[t * input.groups + c / width]
                } else {
                    input.sy[c / width]
                };
                let delta = out[t * input.outputs + c] as f32 * scale - input.physical(t, c);
                max_error = max_error.max(delta.abs());
                squares += (delta as f64) * (delta as f64);
            }
        }
        let raw: Vec<u8> = out.iter().map(|&v| v as u8).collect();
        variants.push(Variant {
            name: name.into(),
            instructions,
            checksum: hash(&raw),
            clamped,
            rmse: (squares / out.len() as f64).sqrt(),
            max_abs_error: max_error,
            different_from_fixed_float: out.iter().zip(&floating).filter(|(a, b)| a != b).count()
                as u64,
        });
    }
    Ok(Report {
        prepare_instructions,
        variants,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_input_rejects_wrong_shape_and_trailing_bytes() {
        let mut bytes = Vec::new();
        for n in [1u32, 1024, 1, 0] {
            bytes.extend(n.to_le_bytes());
        }
        for _ in 0..1024 {
            bytes.extend(1i32.to_le_bytes());
        }
        for _ in 0..1026 {
            bytes.extend(0.5f32.to_le_bytes());
        }
        assert!(Input::decode(&bytes).is_ok());
        assert!(Input::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Input::decode(&trailing).is_err());
        bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(Input::decode(&bytes).is_err());
    }
    #[test]
    fn fixed_q31_matches_float_ties_bias_and_clipping() {
        let input = Input {
            tokens: 2,
            outputs: 4,
            groups: 1,
            sums: vec![1, -1, 3, -3, 511, -511, 0, 0],
            sx: vec![0.5; 2],
            sw: vec![1.; 4],
            sy: vec![1.],
            bias: Some(vec![0., 0., 0., 0.]),
        };
        let p = prepare(&input).unwrap();
        let (a, ca) = fixed_float(&input);
        let (b, cb) = fixed_integer(&input, &p);
        assert_eq!(a, vec![1, -1, 2, -2, 127, -127, 0, 0]);
        assert_eq!(a, b);
        assert_eq!((ca, cb), (2, 2));
        let input = Input {
            bias: Some(vec![0.25, -0.25, 0.5, -0.5]),
            ..input
        };
        let p = prepare(&input).unwrap();
        assert_eq!(fixed_float(&input), fixed_integer(&input, &p));
    }
    #[test]
    fn fixed_q31_checks_range_before_integer_arithmetic() {
        let mut input = Input {
            tokens: 1,
            outputs: 1,
            groups: 1,
            sums: vec![1],
            sx: vec![0.5],
            sw: vec![1.],
            sy: vec![1.],
            bias: None,
        };
        input.validate().unwrap();
        prepare(&input).unwrap();
        input.sx[0] = 2.;
        assert!(prepare(&input).is_err());
        input.sx[0] = 0.5;
        input.bias = Some(vec![1e30]);
        assert!(prepare(&input).is_err());
        input.bias = None;
        input.sums[0] = i32::MIN;
        assert!(input.validate().is_err());
    }
}
