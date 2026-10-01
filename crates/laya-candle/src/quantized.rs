//! Experimental INT8 activation engine. Activations remain signed bytes plus
//! per-row scales between operators and queries. Nonlinear functions and scale
//! calculation use floating-point scalars/row scratch, never a full F32 Tensor.
use crate::{Activation, Attention, CResult, DecisionLayer, EncoderLayer, LayaModel, Linear, Norm};
use ic_laya_core::{Digest, Error, Result, TokenInput, MAX_TOKENS};

#[derive(Clone, Debug)]
pub(crate) struct QTensor {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub groups: usize,
}

impl QTensor {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Self::zeros_grouped(rows, cols, 1)
    }
    pub(crate) fn zeros_grouped(rows: usize, cols: usize, groups: usize) -> Self {
        assert!(groups > 0 && cols % groups == 0);
        Self {
            rows,
            cols,
            values: vec![0; rows * cols],
            scales: vec![1.; rows * groups],
            groups,
        }
    }
    pub(crate) fn set_row(&mut self, r: usize, values: &[f32]) -> CResult<()> {
        if values.len() != self.cols {
            candle_core::bail!("q8 row shape")
        }
        let width = self.cols / self.groups;
        for g in 0..self.groups {
            self.scales[r * self.groups + g] = crate::int8::quantize_row(
                &values[g * width..(g + 1) * width],
                &mut self.values[r * self.cols + g * width..r * self.cols + (g + 1) * width],
            )?;
        }
        Ok(())
    }
    fn value(&self, r: usize, c: usize) -> f32 {
        self.values[r * self.cols + c] as f32
            * self.scales[r * self.groups + c / (self.cols / self.groups)]
    }
    fn map(&self, f: impl Fn(f32) -> f32) -> CResult<Self> {
        let mut out = Self::zeros(self.rows, self.cols);
        let mut scratch = vec![0.; self.cols];
        for r in 0..self.rows {
            for (c, v) in scratch.iter_mut().enumerate() {
                *v = f(self.value(r, c));
            }
            out.set_row(r, &scratch)?;
        }
        Ok(out)
    }
    fn add(&self, rhs: &Self) -> CResult<Self> {
        if self.cols != rhs.cols || (rhs.rows != self.rows && rhs.rows != 1) {
            candle_core::bail!("q8 add shape")
        }
        let mut out = Self::zeros(self.rows, self.cols);
        let mut scratch = vec![0.; self.cols];
        for r in 0..self.rows {
            for (c, v) in scratch.iter_mut().enumerate() {
                *v = self.value(r, c) + rhs.value(if rhs.rows == 1 { 0 } else { r }, c);
            }
            out.set_row(r, &scratch)?;
        }
        Ok(out)
    }
    fn select(&self, ids: &[u32]) -> CResult<Self> {
        let mut out = Self::zeros_grouped(ids.len(), self.cols, self.groups);
        for (r, &id) in ids.iter().enumerate() {
            let i = id as usize;
            if i >= self.rows {
                candle_core::bail!("q8 select index")
            }
            out.values[r * self.cols..(r + 1) * self.cols]
                .copy_from_slice(&self.values[i * self.cols..(i + 1) * self.cols]);
            out.scales[r * self.groups..(r + 1) * self.groups]
                .copy_from_slice(&self.scales[i * self.groups..(i + 1) * self.groups]);
        }
        Ok(out)
    }
}

impl Linear {
    fn q8(&self, x: &QTensor) -> CResult<QTensor> {
        let bias = self.bias.as_ref().map(|b| b.to_vec1::<f32>()).transpose()?;
        self.weight.forward_q8(x, bias.as_deref())
    }
    fn q8_grouped(&self, x: &QTensor, groups: usize) -> CResult<QTensor> {
        let bias = self.bias.as_ref().map(|b| b.to_vec1::<f32>()).transpose()?;
        self.weight.forward_q8_grouped(x, bias.as_deref(), groups)
    }
}

impl Norm {
    fn q8(&self, x: &QTensor) -> CResult<QTensor> {
        crate::profile::measure("q8.norm", [x.rows, x.cols, 0], || {
            if x.groups != 1 {
                candle_core::bail!("q8 norm input grouping")
            }
            let w = self.weight.to_vec1::<f32>()?;
            let b = self.bias.as_ref().map(|b| b.to_vec1::<f32>()).transpose()?;
            let mut out = QTensor::zeros(x.rows, x.cols);
            let mut scratch = vec![0.; x.cols];
            for r in 0..x.rows {
                let row = &x.values[r * x.cols..(r + 1) * x.cols];
                let (sum, squares) = crate::q8_ops::statistics(row);
                let mean = sum as f64 / x.cols as f64;
                let variance = (squares as f64 / x.cols as f64 - mean * mean).max(0.);
                let scale = x.scales[r] as f64;
                let inv = scale / (variance * scale * scale + self.eps).sqrt();
                crate::q8_ops::normalize_affine(row, mean, inv, &w, b.as_deref(), &mut scratch);
                out.set_row(r, &scratch)?;
            }
            Ok(out)
        })
    }
}

// Same erf-based GELU as Candle, evaluated on scalars. No approximation table.
fn gelu(x: f32) -> f32 {
    <candle_core::op::GeluErf as candle_core::op::UnaryOpT>::f32(x)
}

impl Attention {
    fn q8(
        &self,
        x: &QTensor,
        rotary: Option<f64>,
        distance: Option<usize>,
        markers: Option<&[u32]>,
    ) -> CResult<QTensor> {
        let t = x.rows;
        let h = x.cols;
        let d = h / self.heads;
        let y = self.qkv.q8_grouped(x, 3 * self.heads)?;
        let merged = crate::profile::measure("q8.attention.core", [t, h, self.heads], || {
            let ids: Vec<usize> = markers.map_or_else(
                || (0..t).collect(),
                |m| m.iter().map(|&i| i as usize).collect(),
            );
            let mut merged = QTensor::zeros(ids.len(), h);
            // Q/K use per-token, per-head scales. V uses one scale per head over
            // the sequence, allowing both QK and AV to accumulate in INT32.
            let mut scratch = vec![0.; t * d];
            let mut head_outputs = vec![0f32; ids.len() * h];
            let mut scores = vec![0i32; t];
            let mut av = vec![0i32; d];
            let rotations = rotary.map(|theta| {
                (0..t)
                    .flat_map(|pos| {
                        (0..d / 2).map(move |i| {
                            let angle = pos as f64 / theta.powf((2 * i) as f64 / d as f64);
                            let (sin, cos) = angle.sin_cos();
                            [sin as f32, cos as f32]
                        })
                    })
                    .collect::<Vec<_>>()
            });
            // Each head overwrites every entry; reuse query-local scratch buffers.
            let mut arrays = [vec![0i8; t * d], vec![0i8; t * d], vec![0i8; t * d]];
            let mut scales = [vec![1f32; t], vec![1f32; t], vec![1f32; t]];
            let mut vt = vec![0i8; t * d];
            let mut probs = vec![0f32; t];
            let mut qp = vec![0i8; t];
            for head in 0..self.heads {
                for kind in 0..3 {
                    for pos in 0..t {
                        let offset = pos * y.cols + kind * h + head * d;
                        let source = &y.values[offset..offset + d];
                        let scale = y.scales[pos * y.groups + kind * self.heads + head];
                        let target = &mut scratch[pos * d..(pos + 1) * d];
                        if kind < 2 {
                            if let Some(rotations) = &rotations {
                                let half = d / 2;
                                crate::q8_ops::rotary(
                                    source,
                                    scale,
                                    &rotations[pos * half..(pos + 1) * half],
                                    target,
                                );
                            } else {
                                crate::q8_ops::dequantize(source, scale, target);
                            }
                        } else {
                            crate::q8_ops::dequantize(source, scale, target);
                        }
                    }
                    if kind < 2 {
                        for pos in 0..t {
                            scales[kind][pos] = crate::int8::quantize_row(
                                &scratch[pos * d..(pos + 1) * d],
                                &mut arrays[kind][pos * d..(pos + 1) * d],
                            )?;
                        }
                    } else {
                        let scale = crate::int8::quantize_row(&scratch, &mut arrays[kind])?;
                        scales[kind].fill(scale);
                    }
                }
                for pos in 0..t {
                    for c in 0..d {
                        vt[c * t + pos] = arrays[2][pos * d + c];
                    }
                }
                for (r, &pos) in ids.iter().enumerate() {
                    crate::int8::integer_dots(
                        &arrays[0][pos * d..(pos + 1) * d],
                        &arrays[1],
                        &mut scores,
                    );
                    let mut peak = f32::NEG_INFINITY;
                    for k in 0..t {
                        let score = if distance.is_some_and(|dist| pos.abs_diff(k) > dist) {
                            f32::NEG_INFINITY
                        } else {
                            scores[k] as f32 * scales[0][pos] * scales[1][k] / (d as f32).sqrt()
                        };
                        probs[k] = score;
                        peak = peak.max(score);
                    }
                    let mut sum = 0.;
                    for p in &mut probs {
                        *p = (*p - peak).exp();
                        sum += *p;
                    }
                    for p in &mut probs {
                        *p /= sum;
                    }
                    let sp = crate::int8::quantize_row(&probs, &mut qp)?;
                    let scale = sp * scales[2][0];
                    crate::int8::integer_dots(&qp, &vt, &mut av);
                    crate::q8_ops::scale_i32(
                        &av,
                        scale,
                        &mut head_outputs[r * h + head * d..r * h + (head + 1) * d],
                    );
                }
            }
            for r in 0..ids.len() {
                merged.set_row(r, &head_outputs[r * h..(r + 1) * h])?;
            }
            Ok::<_, candle_core::Error>(merged)
        })?;
        self.out.q8(&merged)
    }
}

impl EncoderLayer {
    fn q8(&self, x: &QTensor) -> CResult<QTensor> {
        let normalized = match &self.attention_norm {
            Some(n) => n.q8(x)?,
            None => x.clone(),
        };
        let x = x.add(
            &self
                .attention
                .q8(&normalized, Some(self.theta), self.distance, None)?,
        )?;
        let y = self.wi.q8_grouped(&self.mlp_norm.q8(&x)?, 2)?;
        let half = y.cols / 2;
        let mut gate = QTensor::zeros(y.rows, half);
        let mut scratch = vec![0.; half];
        for r in 0..y.rows {
            for c in 0..half {
                scratch[c] = gelu(y.value(r, c)) * y.value(r, half + c);
            }
            gate.set_row(r, &scratch)?;
        }
        x.add(&self.wo.q8(&gate)?)
    }
}

impl DecisionLayer {
    fn ff_q8(&self, x: &QTensor) -> CResult<QTensor> {
        let x = self.linear1.q8(x)?.map(|v| match self.activation {
            Activation::Relu => v.max(0.),
            Activation::Gelu => gelu(v),
        })?;
        self.linear2.q8(&x)
    }
    fn q8(&self, x: &QTensor, markers: Option<&[u32]>) -> CResult<QTensor> {
        let selected = match markers {
            Some(m) => x.select(m)?,
            None => x.clone(),
        };
        if self.norm_first {
            let x = selected.add(&self.attention.q8(&self.norm1.q8(x)?, None, None, markers)?)?;
            x.add(&self.ff_q8(&self.norm2.q8(&x)?)?)
        } else {
            let x = self
                .norm1
                .q8(&selected.add(&self.attention.q8(x, None, None, markers)?)?)?;
            self.norm2.q8(&x.add(&self.ff_q8(&x)?)?)
        }
    }
}

#[derive(Clone)]
pub struct QuantizedSession {
    bundle: Digest,
    input: TokenInput,
    hidden: QTensor,
    next: usize,
}
impl QuantizedSession {
    pub fn completed_steps(&self) -> usize {
        self.next
    }
}

impl LayaModel {
    pub fn begin_quantized(&self, input: TokenInput) -> Result<QuantizedSession> {
        self.validate_input(&input)?;
        let hidden = self
            .embedding
            .gather_q8(&input.input_ids)
            .and_then(|x| self.embedding_norm.q8(&x))
            .map_err(model_error)?;
        Ok(QuantizedSession {
            bundle: self.bundle,
            input,
            hidden,
            next: 0,
        })
    }
    pub fn step_quantized(&self, s: &mut QuantizedSession) -> Result<Option<Vec<f32>>> {
        if s.bundle != self.bundle {
            return Err(Error::BindingMismatch);
        }
        let n = s.next;
        if n >= self.inference_steps() {
            return Err(Error::Transition);
        }
        let hidden = if n < self.layers.len() {
            self.layers[n].q8(&s.hidden)
        } else if n == self.layers.len() {
            self.final_norm
                .q8(&s.hidden)
                .and_then(|x| x.add(&self.qtype.gather_q8(&[s.input.qtype_id])?))
        } else if n < self.inference_steps() - 1 {
            let i = n - self.layers.len() - 1;
            self.decision[i].q8(
                &s.hidden,
                if i + 1 == self.decision.len() {
                    Some(&s.input.markers)
                } else {
                    None
                },
            )
        } else {
            self.scorer_norm
                .q8(&s.hidden)
                .and_then(|x| self.scorer_dense.q8(&x))
                .and_then(|x| x.map(gelu))
                .and_then(|x| self.scorer_out.q8(&x))
        }
        .map_err(model_error)?;
        let logits = if n == self.inference_steps() - 1 {
            Some(
                (0..hidden.rows)
                    .map(|r| hidden.value(r, 0))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        if logits
            .as_ref()
            .is_some_and(|v| v.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Numeric);
        }
        s.hidden = hidden;
        s.next += 1;
        Ok(logits)
    }
    pub fn export_quantized(&self, s: &QuantizedSession) -> Result<Vec<u8>> {
        if s.bundle != self.bundle {
            return Err(Error::BindingMismatch);
        }
        self.validate_input(&s.input)?;
        let rows = self.q8_rows(&s.input, s.next)?;
        if s.hidden.rows != rows || s.hidden.cols != self.config.hidden_size || s.hidden.groups != 1
        {
            return Err(Error::Invalid("q8 continuation shape".into()));
        }
        let mut bytes = Vec::with_capacity(
            60 + 4 * (s.input.input_ids.len() + s.input.markers.len() + rows)
                + s.hidden.values.len(),
        );
        bytes.extend_from_slice(b"LAYI");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&QUANTIZED_REVISION.to_le_bytes());
        bytes.extend_from_slice(&self.bundle);
        for v in [
            s.next as u32,
            s.input.qtype_id,
            s.input.input_ids.len() as u32,
            s.input.markers.len() as u32,
        ] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in s.input.input_ids.iter().chain(&s.input.markers) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &scale in &s.hidden.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        bytes.extend(s.hidden.values.iter().map(|&v| v as u8));
        Ok(bytes)
    }
    fn q8_rows(&self, input: &TokenInput, next: usize) -> Result<usize> {
        if next >= self.inference_steps() {
            return Err(Error::Transition);
        }
        Ok(if next == self.inference_steps() - 1 {
            input.markers.len()
        } else {
            input.input_ids.len()
        })
    }
    pub fn import_quantized(&self, bytes: &[u8]) -> Result<QuantizedSession> {
        if bytes.len() > MAX_QUANTIZED_BYTES {
            return Err(Error::TooLong);
        }
        if bytes.len() < 60 || &bytes[..4] != b"LAYI" {
            return Err(Error::Invalid("q8 continuation header".into()));
        }
        let read = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        if read(4) != 1 || read(8) != QUANTIZED_REVISION || bytes[12..44] != self.bundle {
            return Err(Error::BindingMismatch);
        }
        let next = read(44) as usize;
        let tokens = read(52) as usize;
        let markers = read(56) as usize;
        if !(1..=MAX_TOKENS).contains(&tokens) || !(2..=7).contains(&markers) {
            return Err(Error::Invalid("q8 input lengths".into()));
        }
        let rows = if next == self.inference_steps() - 1 {
            markers
        } else {
            tokens
        };
        let input_end = 60 + 4 * (tokens + markers);
        let data_start = input_end + 4 * rows;
        if bytes.len() != data_start + rows * self.config.hidden_size {
            return Err(Error::Invalid("q8 continuation length".into()));
        }
        let input = TokenInput {
            input_ids: (0..tokens).map(|i| read(60 + 4 * i)).collect(),
            markers: (0..markers).map(|i| read(60 + 4 * (tokens + i))).collect(),
            qtype_id: read(48),
        };
        self.validate_input(&input)?;
        self.q8_rows(&input, next)?;
        let scales: Vec<f32> = bytes[input_end..data_start]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        // Bound dequantized values too: finite scales alone can still overflow.
        if scales
            .iter()
            .any(|&v| !v.is_finite() || v <= 0. || v > f32::MAX / 128.)
        {
            return Err(Error::Numeric);
        }
        let values = bytes[data_start..].iter().map(|&v| v as i8).collect();
        Ok(QuantizedSession {
            bundle: self.bundle,
            input,
            hidden: QTensor {
                rows,
                cols: self.config.hidden_size,
                values,
                scales,
                groups: 1,
            },
            next,
        })
    }
}

fn model_error(e: candle_core::Error) -> Error {
    Error::ModelUnavailable(e.to_string())
}
pub const MAX_QUANTIZED_BYTES: usize = 60 + 4 * (MAX_TOKENS + 7 + MAX_TOKENS) + MAX_TOKENS * 2048;
pub const QUANTIZED_REVISION: u32 = 2;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_linear_requantization_matches_independent_reference() {
        for cols in [1, 15, 16, 17, 33, 1024] {
            for rows in [1, 4, 17, 48] {
                for tokens in [1, 3, 5, 8, 65] {
                    let w: Vec<i8> = (0..rows * cols)
                        .map(|i| ((i * 37 + 11) % 255) as i32 - 127)
                        .map(|i| i as i8)
                        .collect();
                    let sw: Vec<f32> = (0..rows).map(|i| 0.01 + (i as f32) * 0.002).collect();
                    let mut bytes: Vec<u8> = w.iter().map(|&v| v as u8).collect();
                    for &v in &sw {
                        bytes.extend(v.to_le_bytes());
                    }
                    let matrix = crate::int8::Int8Matrix::from_bytes(&bytes, rows, cols).unwrap();
                    let mut x = QTensor::zeros(tokens, cols);
                    for (i, v) in x.values.iter_mut().enumerate() {
                        *v = ((i * 19) % 255) as i8;
                    }
                    for (i, v) in x.scales.iter_mut().enumerate() {
                        *v = 0.05 + (i as f32) * 0.001;
                    }
                    let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.13 - 0.4).collect();
                    for groups in [1, 2, 3, 4, rows] {
                        if rows % groups != 0 {
                            continue;
                        }
                        let actual = matrix.forward_q8_grouped(&x, Some(&bias), groups).unwrap();
                        let mut expected = QTensor::zeros_grouped(tokens, rows, groups);
                        for t in 0..tokens {
                            let row: Vec<f32> = (0..rows)
                                .map(|r| {
                                    let sum: i32 = (0..cols)
                                        .map(|c| {
                                            (x.values[t * cols + c] as i32)
                                                * (w[r * cols + c] as i32)
                                        })
                                        .sum();
                                    sum as f32 * x.scales[t] * sw[r] + bias[r]
                                })
                                .collect();
                            expected.set_row(t, &row).unwrap();
                        }
                        assert_eq!(actual.values, expected.values);
                        assert_eq!(actual.scales, expected.scales);
                    }
                }
            }
        }
    }
    #[test]
    fn normalization_and_residual_keep_row_scales() {
        let mut x = QTensor::zeros(2, 4);
        x.set_row(0, &[1., 2., 3., 4.]).unwrap();
        x.set_row(1, &[100., -100., 50., -50.]).unwrap();
        let y = x.add(&x).unwrap();
        assert_eq!(y.values, x.values);
        for r in 0..2 {
            assert_eq!(y.scales[r], x.scales[r] * 2.);
        }
        let weight =
            candle_core::Tensor::from_vec(vec![1f32; 4], 4, &candle_core::Device::Cpu).unwrap();
        let norm = Norm::new(weight, None, 1e-5);
        let y = norm.q8(&x).unwrap();
        for r in 0..2 {
            let row: Vec<f32> = (0..4).map(|c| x.value(r, c)).collect();
            let tensor =
                candle_core::Tensor::from_vec(row, (1, 4), &candle_core::Device::Cpu).unwrap();
            let expected = norm.forward(&tensor).unwrap().to_vec2::<f32>().unwrap();
            for c in 0..4 {
                assert!((y.value(r, c) - expected[0][c]).abs() < 0.02);
            }
        }
    }
}
