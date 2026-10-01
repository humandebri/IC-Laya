//! Lossless, bounded client-held continuation format. This is not an authenticated receipt.
use crate::{InferenceSession, LayaModel};
use candle_core::{Device, Tensor};
use ic_laya_core::{Error, Result, TokenInput, MAX_TOKENS};

pub const FORMAT_VERSION: u32 = 1;
/// Bump when inference semantics or phase boundaries become incompatible.
pub const INFERENCE_REVISION: u32 = 1;
const HEADER_BYTES: usize = 60;
pub const MAX_CONTINUATION_BYTES: usize =
    HEADER_BYTES + MAX_TOKENS * 4 + 7 * 4 + MAX_TOKENS * 2048 * 4;

impl LayaModel {
    fn continuation_shape(&self, input: &TokenInput, next: usize) -> Result<(usize, usize)> {
        self.validate_input(input)?;
        if next >= self.inference_steps() {
            return Err(Error::Transition);
        }
        // After the final decision phase only marker rows remain, ready for the scorer.
        let rows = if next == self.inference_steps() - 1 {
            input.markers.len()
        } else {
            input.input_ids.len()
        };
        Ok((rows, self.config.hidden_size))
    }

    pub fn export_inference(&self, session: &InferenceSession) -> Result<Vec<u8>> {
        if session.bundle != self.bundle {
            return Err(Error::BindingMismatch);
        }
        let shape = self.continuation_shape(&session.input, session.next)?;
        if session.hidden.dims() != [shape.0, shape.1] {
            return Err(Error::Invalid("continuation shape".into()));
        }
        let values = session
            .hidden
            .flatten_all()
            .and_then(|x| x.to_vec1::<f32>())
            .map_err(|e| Error::ModelUnavailable(e.to_string()))?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numeric);
        }
        let size = HEADER_BYTES
            + (session.input.input_ids.len() + session.input.markers.len() + values.len()) * 4;
        if size > MAX_CONTINUATION_BYTES {
            return Err(Error::TooLong);
        }
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(b"LAYQ");
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&INFERENCE_REVISION.to_le_bytes());
        bytes.extend_from_slice(&self.bundle);
        for v in [
            session.next as u32,
            session.input.qtype_id,
            session.input.input_ids.len() as u32,
            session.input.markers.len() as u32,
        ] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in session.input.input_ids.iter().chain(&session.input.markers) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(bytes)
    }

    pub fn import_inference(&self, bytes: &[u8]) -> Result<InferenceSession> {
        if bytes.len() > MAX_CONTINUATION_BYTES {
            return Err(Error::TooLong);
        }
        if bytes.len() < HEADER_BYTES || &bytes[..4] != b"LAYQ" {
            return Err(Error::Invalid("continuation header".into()));
        }
        // All fixed offsets are safe after the header length check.
        let read = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if read(4) != FORMAT_VERSION
            || read(8) != INFERENCE_REVISION
            || bytes[12..44] != self.bundle
        {
            return Err(Error::BindingMismatch);
        }
        let next = read(44) as usize;
        let qtype_id = read(48);
        let tokens = read(52) as usize;
        let markers = read(56) as usize;
        if !(1..=MAX_TOKENS).contains(&tokens) || !(2..=7).contains(&markers) {
            return Err(Error::Invalid("continuation input lengths".into()));
        }
        if next >= self.inference_steps() {
            return Err(Error::Transition);
        }
        let rows = if next == self.inference_steps() - 1 {
            markers
        } else {
            tokens
        };
        let elements = rows
            .checked_mul(self.config.hidden_size)
            .ok_or(Error::TooLong)?;
        let input_end = HEADER_BYTES + (tokens + markers) * 4;
        let expected = elements
            .checked_mul(4)
            .and_then(|n| input_end.checked_add(n))
            .ok_or(Error::TooLong)?;
        if bytes.len() != expected {
            return Err(Error::Invalid("continuation byte length".into()));
        }
        let input = TokenInput {
            input_ids: (0..tokens).map(|i| read(HEADER_BYTES + i * 4)).collect(),
            markers: (0..markers)
                .map(|i| read(HEADER_BYTES + (tokens + i) * 4))
                .collect(),
            qtype_id,
        };
        self.continuation_shape(&input, next)?;
        let mut values = Vec::with_capacity(elements);
        for chunk in bytes[input_end..].chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            if !value.is_finite() {
                return Err(Error::Numeric);
            }
            values.push(value);
        }
        let hidden = Tensor::from_vec(values, (rows, self.config.hidden_size), &Device::Cpu)
            .map_err(|e| Error::ModelUnavailable(e.to_string()))?;
        Ok(InferenceSession {
            bundle: self.bundle,
            input,
            hidden,
            next,
        })
    }
}
