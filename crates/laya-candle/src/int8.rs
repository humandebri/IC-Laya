//! Symmetric per-output-row W8A8. Activations are quantized per token;
//! accumulation is i32, while scales, bias, norms and attention stay F32.
use candle_core::{CpuStorage, Device, Storage, Tensor};
use std::borrow::Cow;
use std::sync::Arc;
#[path = "int8_quant.rs"]
mod quant;

#[derive(Clone)]
pub struct Int8Matrix {
    rows: usize,
    cols: usize,
    values: Arc<[i8]>,
    scales: Arc<[f32]>,
}
impl Int8Matrix {
    /// Integer dot products from already quantized activations. The result is
    /// requantized row by row; no full F32 activation tensor is materialized.
    pub(crate) fn forward_q8(
        &self,
        x: &crate::quantized::QTensor,
        bias: Option<&[f32]>,
    ) -> candle_core::Result<crate::quantized::QTensor> {
        self.forward_q8_grouped(x, bias, 1)
    }
    pub(crate) fn forward_q8_grouped(
        &self,
        x: &crate::quantized::QTensor,
        bias: Option<&[f32]>,
        groups: usize,
    ) -> candle_core::Result<crate::quantized::QTensor> {
        if x.cols != self.cols
            || x.groups != 1
            || groups == 0
            || self.rows % groups != 0
            || bias.is_some_and(|b| b.len() != self.rows)
        {
            candle_core::bail!("q8 linear shape")
        }
        crate::profile::measure("q8.linear", [x.rows, self.rows, self.cols], || {
            let mut result = crate::quantized::QTensor::zeros_grouped(x.rows, self.rows, groups);
            let mut t = 0;
            while t < x.rows {
                macro_rules! tile {
                    ($r:literal) => {{
                        self.q8_tile::<$r>(x, bias, t, &mut result)?;
                        t += $r;
                    }};
                }
                let left = x.rows - t;
                if left >= 64 {
                    tile!(64);
                } else if left >= 32 {
                    tile!(32);
                } else if left >= 16 {
                    tile!(16);
                } else if left >= 8 {
                    tile!(8);
                } else if left >= 4 {
                    tile!(4);
                } else if left >= 2 {
                    tile!(2);
                } else {
                    tile!(1);
                }
            }
            Ok(result)
        })
    }

    fn q8_tile<const R: usize>(
        &self,
        x: &crate::quantized::QTensor,
        bias: Option<&[f32]>,
        t: usize,
        out: &mut crate::quantized::QTensor,
    ) -> candle_core::Result<()> {
        let a = &x.values[t * self.cols..(t + R) * self.cols];
        let width = self.rows / out.groups;
        // A single reusable output-group scratch buffer. INT32 dot tiles feed
        // scale+bias directly, without a separate R*outputs INT32 allocation.
        let mut block = vec![0f32; R * width];
        for group in 0..out.groups {
            let base = group * width;
            crate::profile::measure("q8.fused_dot_writeback", [R, width, self.cols], || {
                let mut r = 0;
                while r < width {
                    macro_rules! dot {
                        ($c:literal) => {{
                            let sums = dot_tile::<R, $c>(
                                a,
                                &self.values[(base + r) * self.cols..(base + r + $c) * self.cols],
                                self.cols,
                            );
                            q8_store_tile::<R, $c>(
                                &sums,
                                &x.scales[t..t + R],
                                &self.scales[base + r..base + r + $c],
                                bias.map(|b| &b[base + r..base + r + $c]),
                                r,
                                width,
                                &mut block,
                            );
                            r += $c;
                        }};
                    }
                    if r + 16 <= width {
                        dot!(16);
                    } else if r + 8 <= width {
                        dot!(8);
                    } else if r + 4 <= width {
                        dot!(4);
                    } else {
                        dot!(1);
                    }
                }
            });
            crate::profile::measure("q8.output_quantize", [R, width, self.cols], || {
                for i in 0..R {
                    out.scales[(t + i) * out.groups + group] = quant::row(
                        &block[i * width..(i + 1) * width],
                        &mut out.values
                            [(t + i) * self.rows + base..(t + i) * self.rows + base + width],
                    )?;
                }
                Ok::<_, candle_core::Error>(())
            })?;
        }
        Ok(())
    }

    pub(crate) fn gather_q8(&self, ids: &[u32]) -> candle_core::Result<crate::quantized::QTensor> {
        let mut out = crate::quantized::QTensor::zeros(ids.len(), self.cols);
        for (i, &id) in ids.iter().enumerate() {
            let r = id as usize;
            if r >= self.rows {
                candle_core::bail!("q8 embedding index")
            }
            out.values[i * self.cols..(i + 1) * self.cols]
                .copy_from_slice(&self.values[r * self.cols..(r + 1) * self.cols]);
            out.scales[i] = self.scales[r];
        }
        Ok(out)
    }
    pub fn from_bytes(bytes: &[u8], rows: usize, cols: usize) -> candle_core::Result<Self> {
        // 16384 * 128 * 128 fits i32, including hostile -128 values.
        if rows == 0 || cols == 0 || cols > 16384 {
            candle_core::bail!("invalid int8 dimensions")
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| candle_core::Error::Msg("int8 size overflow".into()))?;
        let length = rows
            .checked_mul(4)
            .and_then(|n| count.checked_add(n))
            .ok_or_else(|| candle_core::Error::Msg("int8 size overflow".into()))?;
        if bytes.len() != length {
            candle_core::bail!("invalid int8 payload length")
        }
        let scales: Vec<f32> = bytes[count..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
            candle_core::bail!("invalid int8 scale")
        }
        Ok(Self {
            rows,
            cols,
            values: bytes[..count].iter().map(|b| *b as i8).collect(),
            scales: scales.into(),
        })
    }
    pub fn dims(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
    pub fn gather(&self, ids: &[u32]) -> candle_core::Result<Tensor> {
        let mut out = Vec::with_capacity(ids.len() * self.cols);
        for &id in ids {
            let row = id as usize;
            if row >= self.rows {
                candle_core::bail!("int8 embedding index")
            }
            out.extend(
                self.values[row * self.cols..(row + 1) * self.cols]
                    .iter()
                    .map(|v| *v as f32 * self.scales[row]),
            );
        }
        Tensor::from_vec(out, (ids.len(), self.cols), &Device::Cpu)
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (tokens, cols) = x.dims2()?;
        if cols != self.cols {
            candle_core::bail!("int8 linear shape")
        }
        let shape = [tokens, self.rows, cols];
        let (storage, layout) = x.storage_and_layout();
        let input = crate::profile::measure("int8.input", shape, || {
            // Quantization only reads the input. Borrow contiguous CPU storage,
            // including a narrowed tensor's offset, instead of copying all F32s.
            match (&*storage, layout.contiguous_offsets()) {
                (Storage::Cpu(CpuStorage::F32(values)), Some((start, end))) => {
                    Ok(Cow::Borrowed(&values[start..end]))
                }
                _ => x.flatten_all()?.to_vec1::<f32>().map(Cow::Owned),
            }
        })?;
        let (activation, scales) = crate::profile::measure("int8.quantize", shape, || {
            let mut activation = vec![0i8; tokens * cols];
            let mut scales = Vec::with_capacity(tokens);
            for (t, row) in input.chunks_exact(cols).enumerate() {
                let scale = quant::row(row, &mut activation[t * cols..(t + 1) * cols])?;
                scales.push(scale);
            }
            Ok::<_, candle_core::Error>((activation, scales))
        })?;
        drop(input);
        drop(storage);
        let out = crate::profile::measure("int8.matmul", shape, || {
            matmul(&activation, &scales, self, tokens)
        });
        crate::profile::measure("int8.tensor", shape, || {
            Tensor::from_vec(out, (tokens, self.rows), x.device())
        })
    }
}

pub(crate) fn quantize_row(input: &[f32], out: &mut [i8]) -> candle_core::Result<f32> {
    quant::row(input, out)
}

#[cfg(not(target_arch = "wasm32"))]
fn q8_store_tile<const R: usize, const C: usize>(
    sums: &[[i32; C]; R],
    sx: &[f32],
    sw: &[f32],
    bias: Option<&[f32]>,
    r: usize,
    width: usize,
    out: &mut [f32],
) {
    for i in 0..R {
        for j in 0..C {
            out[i * width + r + j] = sums[i][j] as f32 * sx[i] * sw[j] + bias.map_or(0., |b| b[j]);
        }
    }
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
fn q8_store_tile<const R: usize, const C: usize>(
    sums: &[[i32; C]; R],
    sx: &[f32],
    sw: &[f32],
    bias: Option<&[f32]>,
    r: usize,
    width: usize,
    out: &mut [f32],
) {
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn kernel<const R: usize, const C: usize>(
        sums: &[[i32; C]; R],
        sx: &[f32],
        sw: &[f32],
        bias: Option<&[f32]>,
        r: usize,
        width: usize,
        out: &mut [f32],
    ) {
        for i in 0..R {
            // Slices are checked before any pointer access; the tile dispatcher
            // provides C weight/bias entries and R activation scales.
            let row = &mut out[i * width + r..i * width + r + C];
            let weights = &sw[..C];
            let bias = bias.map(|b| &b[..C]);
            let scale = f32x4_splat(sx[i]);
            let mut j = 0;
            while j + 4 <= C {
                let sum = unsafe { v128_load(sums[i].as_ptr().add(j).cast()) };
                let sw = unsafe { v128_load(weights.as_ptr().add(j).cast()) };
                let y = f32x4_mul(f32x4_mul(f32x4_convert_i32x4(sum), scale), sw);
                let b = match bias {
                    Some(b) => unsafe { v128_load(b.as_ptr().add(j).cast()) },
                    None => f32x4_splat(0.),
                };
                unsafe { v128_store(row.as_mut_ptr().add(j).cast(), f32x4_add(y, b)) };
                j += 4;
            }
            while j < C {
                row[j] = sums[i][j] as f32 * sx[i] * weights[j] + bias.map_or(0., |b| b[j]);
                j += 1;
            }
        }
    }
    // Every complete vector chunk is bounded by C; checked slices bound each row.
    unsafe { kernel::<R, C>(sums, sx, sw, bias, r, width, out) }
}
/// Compute many independent dots sharing one input. Each complete output tile
/// reuses its input load/sign-extension; sums remain exact bounded INT32.
pub(crate) fn integer_dots(a: &[i8], b: &[i8], out: &mut [i32]) {
    assert!(!a.is_empty() && a.len()<=16384 && b.len()==a.len()*out.len());
    let mut c=0;
    while c<out.len() {
        macro_rules! tile { ($n:literal) => {{
            let sums=dot_tile::<1,$n>(a,&b[c*a.len()..(c+$n)*a.len()],a.len());
            out[c..c+$n].copy_from_slice(&sums[0]);c+=$n;
        }}; }
        if c+16<=out.len() { tile!(16); }
        else if c+8<=out.len() { tile!(8); }
        else if c+4<=out.len() { tile!(4); }
        else { tile!(1); }
    }
}

#[cfg(test)]
pub(crate) fn integer_dot(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    dot_tile::<1, 1>(a, b, a.len())[0][0]
}

const MAX_TILE_ROWS: usize = 64;
const MAX_TILE_COLS: usize = 16;
const PAD_TAIL: bool = true;

// Padding exists only inside this linear operator, never in the attention sequence.
fn matmul(activation: &[i8], scales: &[f32], weight: &Int8Matrix, tokens: usize) -> Vec<f32> {
    let cols = weight.cols;
    let padded_tokens = if PAD_TAIL && tokens % 4 == 3 {
        tokens.div_ceil(4) * 4
    } else {
        tokens
    };
    let padded;
    let (a, scales) = if padded_tokens != tokens {
        let mut input = activation.to_vec();
        input.resize(padded_tokens * cols, 0);
        let mut sx = scales.to_vec();
        sx.resize(padded_tokens, 1.);
        padded = (input, sx);
        (&padded.0[..], &padded.1[..])
    } else {
        (activation, scales)
    };
    let mut out = vec![0f32; padded_tokens * weight.rows];
    let mut t = 0;
    while t < padded_tokens {
        let remaining = padded_tokens - t;
        if MAX_TILE_ROWS >= 64 && remaining >= 64 {
            write_tile::<64>(a, scales, weight, t, &mut out);
            t += 64;
        } else if MAX_TILE_ROWS >= 32 && remaining >= 32 {
            write_tile::<32>(a, scales, weight, t, &mut out);
            t += 32;
        } else if MAX_TILE_ROWS >= 16 && remaining >= 16 {
            write_tile::<16>(a, scales, weight, t, &mut out);
            t += 16;
        } else if MAX_TILE_ROWS >= 8 && remaining >= 8 {
            write_tile::<8>(a, scales, weight, t, &mut out);
            t += 8;
        } else if remaining >= 4 {
            write_tile::<4>(a, scales, weight, t, &mut out);
            t += 4;
        } else if remaining >= 2 {
            write_tile::<2>(a, scales, weight, t, &mut out);
            t += 2;
        } else {
            write_tile::<1>(a, scales, weight, t, &mut out);
            t += 1;
        }
    }
    out.truncate(tokens * weight.rows);
    out
}
fn write_tile<const R: usize>(a: &[i8], sx: &[f32], w: &Int8Matrix, t: usize, out: &mut [f32]) {
    let a = &a[t * w.cols..(t + R) * w.cols];
    let mut r = 0;
    while r < w.rows {
        macro_rules! write {
            ($c:literal) => {{
                let sums = dot_tile::<R, $c>(a, &w.values[r * w.cols..(r + $c) * w.cols], w.cols);
                store_tile::<R, $c>(&sums, sx, w, t, r, out);
                r += $c;
            }};
        }
        if MAX_TILE_COLS >= 16 && r + 16 <= w.rows {
            write!(16);
        } else if MAX_TILE_COLS >= 8 && r + 8 <= w.rows {
            write!(8);
        } else if r + 4 <= w.rows {
            write!(4);
        } else {
            write!(1);
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn store_tile<const R: usize, const C: usize>(
    sums: &[[i32; C]; R],
    sx: &[f32],
    w: &Int8Matrix,
    t: usize,
    r: usize,
    out: &mut [f32],
) {
    for i in 0..R {
        for j in 0..C {
            out[(t + i) * w.rows + r + j] = sums[i][j] as f32 * sx[t + i] * w.scales[r + j];
        }
    }
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
fn store_tile<const R: usize, const C: usize>(
    sums: &[[i32; C]; R],
    sx: &[f32],
    w: &Int8Matrix,
    t: usize,
    r: usize,
    out: &mut [f32],
) {
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn kernel<const R: usize, const C: usize>(
        sums: &[[i32; C]; R],
        sx: &[f32],
        w: &Int8Matrix,
        t: usize,
        r: usize,
        out: &mut [f32],
    ) {
        for i in 0..R {
            let scale = f32x4_splat(sx[t + i]);
            let row = &mut out[(t + i) * w.rows + r..(t + i) * w.rows + r + C];
            let mut j = 0;
            while j + 4 <= C {
                // Complete four-lane chunks are bounded by the row, sums and scales slices.
                let dots = unsafe { v128_load(sums[i].as_ptr().add(j).cast()) };
                let weights = unsafe { v128_load(w.scales.as_ptr().add(r + j).cast()) };
                let value = f32x4_mul(f32x4_mul(f32x4_convert_i32x4(dots), scale), weights);
                unsafe { v128_store(row.as_mut_ptr().add(j).cast(), value) };
                j += 4;
            }
            while j < C {
                row[j] = sums[i][j] as f32 * sx[t + i] * w.scales[r + j];
                j += 1;
            }
        }
    }
    // Every row and weight tile is checked by write_tile's dispatch.
    unsafe { kernel(sums, sx, w, t, r, out) }
}
#[cfg(not(target_arch = "wasm32"))]
fn dot_tile<const R: usize, const C: usize>(a: &[i8], b: &[i8], cols: usize) -> [[i32; C]; R] {
    let mut result = [[0; C]; R];
    for i in 0..R {
        for j in 0..C {
            for k in 0..cols {
                result[i][j] += a[i * cols + k] as i32 * b[j * cols + k] as i32;
            }
        }
    }
    result
}
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
// K<=16384 bounds every signed i8 dot below 16384*128*128 < i32::MAX.
fn dot_tile<const R: usize, const C: usize>(a: &[i8], b: &[i8], cols: usize) -> [[i32; C]; R] {
    assert!(R <= 64 && C <= 16 && cols <= 16384 && a.len() == R * cols && b.len() == C * cols);
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn kernel<const R: usize, const C: usize>(
        a: &[i8],
        b: &[i8],
        cols: usize,
    ) -> [[i32; C]; R] {
        let mut acc = [[i32x4_splat(0); C]; R];
        let end = cols / 16 * 16;
        let mut k = 0;
        while k + 16 < end {
            {
                let offset = k;
                let mut lo = [i32x4_splat(0); C];
                let mut hi = [i32x4_splat(0); C];
                macro_rules! weights {($($c:literal),*)=>{$(if C>$c{
                // Each full chunk is within the wrapper-checked C rows.
                let w=unsafe{v128_load(b.as_ptr().add($c*cols+offset).cast())};
                lo[$c]=i16x8_extend_low_i8x16(w);hi[$c]=i16x8_extend_high_i8x16(w);
            })*};}
                weights!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                macro_rules! columns {($r:literal,$xl:ident,$xh:ident;$($c:literal),*)=>{$(if C>$c{
                acc[$r][$c]=i32x4_add(acc[$r][$c],i32x4_add(i32x4_dot_i16x8($xl,lo[$c]),i32x4_dot_i16x8($xh,hi[$c])));
            })*};}
                macro_rules! rows {($($r:literal),*)=>{$(if R>$r{
                // Each full chunk is within the wrapper-checked R rows.
                let x=unsafe{v128_load(a.as_ptr().add($r*cols+offset).cast())};
                let xl=i16x8_extend_low_i8x16(x);let xh=i16x8_extend_high_i8x16(x);
                columns!($r,xl,xh;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
            })*};}
                rows!(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                    62, 63
                );
            }
            {
                let offset = k + 16;
                let mut lo = [i32x4_splat(0); C];
                let mut hi = [i32x4_splat(0); C];
                macro_rules! weights {($($c:literal),*)=>{$(if C>$c{
                // Each full chunk is within the wrapper-checked C rows.
                let w=unsafe{v128_load(b.as_ptr().add($c*cols+offset).cast())};
                lo[$c]=i16x8_extend_low_i8x16(w);hi[$c]=i16x8_extend_high_i8x16(w);
            })*};}
                weights!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                macro_rules! columns {($r:literal,$xl:ident,$xh:ident;$($c:literal),*)=>{$(if C>$c{
                acc[$r][$c]=i32x4_add(acc[$r][$c],i32x4_add(i32x4_dot_i16x8($xl,lo[$c]),i32x4_dot_i16x8($xh,hi[$c])));
            })*};}
                macro_rules! rows {($($r:literal),*)=>{$(if R>$r{
                // Each full chunk is within the wrapper-checked R rows.
                let x=unsafe{v128_load(a.as_ptr().add($r*cols+offset).cast())};
                let xl=i16x8_extend_low_i8x16(x);let xh=i16x8_extend_high_i8x16(x);
                columns!($r,xl,xh;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
            })*};}
                rows!(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                    62, 63
                );
            }
            k += 32;
        }
        if k < end {
            let offset = k;
            let mut lo = [i32x4_splat(0); C];
            let mut hi = [i32x4_splat(0); C];
            macro_rules! weights {($($c:literal),*)=>{$(if C>$c{
                // Each full chunk is within the wrapper-checked C rows.
                let w=unsafe{v128_load(b.as_ptr().add($c*cols+offset).cast())};
                lo[$c]=i16x8_extend_low_i8x16(w);hi[$c]=i16x8_extend_high_i8x16(w);
            })*};}
            weights!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            macro_rules! columns {($r:literal,$xl:ident,$xh:ident;$($c:literal),*)=>{$(if C>$c{
                acc[$r][$c]=i32x4_add(acc[$r][$c],i32x4_add(i32x4_dot_i16x8($xl,lo[$c]),i32x4_dot_i16x8($xh,hi[$c])));
            })*};}
            macro_rules! rows {($($r:literal),*)=>{$(if R>$r{
                // Each full chunk is within the wrapper-checked R rows.
                let x=unsafe{v128_load(a.as_ptr().add($r*cols+offset).cast())};
                let xl=i16x8_extend_low_i8x16(x);let xh=i16x8_extend_high_i8x16(x);
                columns!($r,xl,xh;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
            })*};}
            rows!(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
            );
        }
        let mut result = [[0; C]; R];
        macro_rules! store_cols {($r:literal;$($c:literal),*)=>{$(if C>$c{
            let v=acc[$r][$c];result[$r][$c]=i32x4_extract_lane::<0>(v)+i32x4_extract_lane::<1>(v)+i32x4_extract_lane::<2>(v)+i32x4_extract_lane::<3>(v);
        })*};}
        macro_rules! store_rows {($($r:literal),*)=>{$(if R>$r{store_cols!($r;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);})*};}
        store_rows!(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        );
        for i in 0..R {
            for j in 0..C {
                for k in end..cols {
                    result[i][j] += a[i * cols + k] as i32 * b[j * cols + k] as i32;
                }
            }
        }
        result
    }
    // Shapes and accumulator bounds checked above; IC supports simd128.
    unsafe { kernel::<R, C>(a, b, cols) }
}
/// Synthetic profiling helper used by the owner-only bounded canister benchmark.
pub fn benchmark(
    tokens: usize,
    rows: usize,
    cols: usize,
    counter: fn() -> u64,
) -> candle_core::Result<(u64, [u8; 32], Vec<crate::profile::Cost>)> {
    if !(1..=128).contains(&tokens)
        || ![1024, 3072, 5248].contains(&rows)
        || ![1024, 2624].contains(&cols)
    {
        candle_core::bail!("benchmark shape")
    }
    let mut bytes: Vec<u8> = (0..rows * cols)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    for _ in 0..rows {
        bytes.extend(0.25f32.to_le_bytes());
    }
    let w = Int8Matrix::from_bytes(&bytes, rows, cols)?;
    let values: Vec<f32> = (0..tokens * cols)
        .map(|i| ((i * 19 % 255) as i32 - 127) as f32)
        .collect();
    let input = Tensor::from_vec(values, (tokens, cols), &Device::Cpu)?;
    let start = counter();
    let (result, costs) = crate::profile::capture(counter, || w.forward(&input));
    let instructions = counter().saturating_sub(start);
    let output = result?.flatten_all()?.to_vec1::<f32>()?;
    let raw: Vec<u8> = output.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok((instructions, ic_laya_core::hash(&raw), costs))
}

/// Diagnostic-only 128-token kernel using the selected weight layout, tile
/// sizes, dot function and SIMD writeback. Counter calls still perturb timing.
pub fn benchmark_components(
    rows: usize,
    cols: usize,
    counter: fn() -> u64,
) -> candle_core::Result<(u64, u64, u64, [u8; 32])> {
    if ![1024, 3072, 5248].contains(&rows) || ![1024, 2624].contains(&cols) {
        candle_core::bail!("benchmark shape")
    }
    const TOKENS: usize = 128;
    let mut bytes: Vec<u8> = (0..rows * cols)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    for _ in 0..rows {
        bytes.extend(0.25f32.to_le_bytes());
    }
    let weight = Int8Matrix::from_bytes(&bytes, rows, cols)?;
    let input: Vec<f32> = (0..TOKENS * cols)
        .map(|i| ((i * 19 % 255) as i32 - 127) as f32)
        .collect();
    let mut activation = vec![0i8; input.len()];
    let mut scales = Vec::with_capacity(TOKENS);
    for (t, row) in input.chunks_exact(cols).enumerate() {
        scales.push(quant::row(row, &mut activation[t * cols..(t + 1) * cols])?);
    }
    let mut out = vec![0f32; TOKENS * rows];
    let mut dots = 0u64;
    let mut writeback = 0u64;
    let start = counter();
    fn measured_tile<const R: usize>(
        activation: &[i8],
        scales: &[f32],
        weight: &Int8Matrix,
        t: usize,
        out: &mut [f32],
        counter: fn() -> u64,
        dots: &mut u64,
        writeback: &mut u64,
    ) {
        let cols = weight.cols;
        let a = &activation[t * cols..(t + R) * cols];
        for r in (0..weight.rows).step_by(16) {
            let before = counter();
            let sums = dot_tile::<R, 16>(a, &weight.values[r * cols..(r + 16) * cols], cols);
            *dots += counter().saturating_sub(before);
            let before = counter();
            store_tile::<R, 16>(&sums, scales, weight, t, r, out);
            *writeback += counter().saturating_sub(before);
        }
    }
    for t in (0..TOKENS).step_by(MAX_TILE_ROWS) {
        if MAX_TILE_ROWS == 64 {
            measured_tile::<64>(
                &activation,
                &scales,
                &weight,
                t,
                &mut out,
                counter,
                &mut dots,
                &mut writeback,
            );
        } else if MAX_TILE_ROWS == 32 {
            measured_tile::<32>(
                &activation,
                &scales,
                &weight,
                t,
                &mut out,
                counter,
                &mut dots,
                &mut writeback,
            );
        } else {
            measured_tile::<16>(
                &activation,
                &scales,
                &weight,
                t,
                &mut out,
                counter,
                &mut dots,
                &mut writeback,
            );
        }
    }
    let total = counter().saturating_sub(start);
    let raw: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok((total, dots, writeback, ic_laya_core::hash(&raw)))
}

#[cfg(test)]
mod batch_tests {
    #[test]
    fn shared_input_dots_match_i64_reference_with_signed_edges_and_tails() {
        for cols in [1,15,16,17,33,64,128,16384] {
            for outputs in [1,3,4,5,8,17,32,65,128] {
                for pattern in 0..4 {
                    let a:Vec<i8>=(0..cols).map(|i| match pattern {0=>-128,1=>127,2=>((i*37+11)%256) as u8 as i8,_=>0}).collect();
                    let b:Vec<i8>=(0..cols*outputs).map(|i| if pattern==0 {-128} else {((i*19+3)%256) as u8 as i8}).collect();
                    let mut out=vec![0;outputs];super::integer_dots(&a,&b,&mut out);
                    for (j,&actual) in out.iter().enumerate() {
                        let row=&b[j*cols..(j+1)*cols];
                        let expected:i64=a.iter().zip(row).map(|(&a,&b)|a as i64*b as i64).sum();
                        assert_eq!(actual as i64,expected);
                        assert_eq!(actual,super::integer_dot(&a,row));
                    }
                }
            }
        }
    }
}
