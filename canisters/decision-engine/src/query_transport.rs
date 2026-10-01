//! Lossless wire envelope. Limits apply both before and after decompression.
use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use ic_laya_core::{Error, Result};
use std::borrow::Cow;
use std::io::{Read, Write};

const HEADER: usize = 16;
const MAX: usize = laya_candle::continuation::MAX_CONTINUATION_BYTES;

fn invalid() -> Error {
    Error::Invalid("query transport envelope".into())
}

fn float_offset(raw: &[u8]) -> Result<usize> {
    if raw.len() < 60 || &raw[..4] != b"LAYQ" {
        return Err(invalid());
    }
    let tokens = u32::from_le_bytes(raw[52..56].try_into().unwrap()) as usize;
    let markers = u32::from_le_bytes(raw[56..60].try_into().unwrap()) as usize;
    if !(1..=128).contains(&tokens) || !(2..=7).contains(&markers) {
        return Err(invalid());
    }
    let offset = 60 + 4 * (tokens + markers);
    if offset > raw.len() || (raw.len() - offset) % 4 != 0 {
        return Err(invalid());
    }
    Ok(offset)
}

fn shuffle(raw: &[u8], inverse: bool) -> Result<Vec<u8>> {
    let offset = float_offset(raw)?;
    let n = (raw.len() - offset) / 4;
    let mut result = vec![0; raw.len()];
    result[..offset].copy_from_slice(&raw[..offset]);
    for lane in 0..4 {
        for i in 0..n {
            if inverse {
                result[offset + 4 * i + lane] = raw[offset + lane * n + i];
            } else {
                result[offset + lane * n + i] = raw[offset + 4 * i + lane];
            }
        }
    }
    Ok(result)
}

pub fn pack(raw: Vec<u8>) -> Result<Vec<u8>> {
    if raw.len() > MAX || raw.len() < 60 {
        return Err(invalid());
    }
    let codec = match &raw[..4] {
        b"LAYQ" => 2u32,
        b"LAYI" => 1u32,
        _ => return Err(invalid()),
    };
    let prepared = if codec == 2 { Cow::Owned(shuffle(&raw, false)?) } else { Cow::Borrowed(raw.as_slice()) };
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&prepared).map_err(|_| invalid())?;
    let compressed = encoder.finish().map_err(|_| invalid())?;
    // Incompressible input remains in the original, already bounded format.
    if compressed.len() + HEADER >= raw.len() {
        return Ok(raw);
    }
    let mut result = Vec::with_capacity(HEADER + compressed.len());
    result.extend_from_slice(b"LAYZ");
    for value in [1, codec, raw.len() as u32] {
        result.extend_from_slice(&value.to_le_bytes());
    }
    result.extend_from_slice(&compressed);
    Ok(result)
}

pub fn unpack(wire: &[u8]) -> Result<Cow<'_, [u8]>> {
    if wire.len() > MAX {
        return Err(Error::TooLong);
    }
    if !wire.starts_with(b"LAYZ") {
        return Ok(Cow::Borrowed(wire));
    }
    if wire.len() < HEADER {
        return Err(invalid());
    }
    let read = |i| u32::from_le_bytes(wire[i..i + 4].try_into().unwrap());
    let codec = read(8);
    let size = read(12) as usize;
    if read(4) != 1 || !matches!(codec, 1 | 2) || !(60..=MAX).contains(&size) {
        return Err(invalid());
    }
    let mut decoder = ZlibDecoder::new(&wire[HEADER..]);
    let mut raw = Vec::with_capacity(size);
    decoder.by_ref().take((size + 1) as u64).read_to_end(&mut raw).map_err(|_| invalid())?;
    if raw.len() != size || decoder.total_in() as usize != wire.len() - HEADER {
        return Err(invalid());
    }
    if codec == 2 {
        raw = shuffle(&raw, true)?;
    } else if !raw.starts_with(b"LAYI") {
        return Err(invalid());
    }
    Ok(Cow::Owned(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_round_trip_and_malformed_envelopes() {
        for magic in [b"LAYQ", b"LAYI"] {
            let mut raw = vec![0; 60 + 4 * 18 + 16 * 1024 * 4];
            raw[..4].copy_from_slice(magic);
            raw[52..56].copy_from_slice(&16u32.to_le_bytes());
            raw[56..60].copy_from_slice(&2u32.to_le_bytes());
            for (i, value) in raw[132..].iter_mut().enumerate() { *value = (i % 23) as u8; }
            let packed = pack(raw.clone()).unwrap();
            assert!(packed.starts_with(b"LAYZ"));
            assert_eq!(unpack(&packed).unwrap().as_ref(), raw);
            for (offset, value) in [(4, 2), (8, 3), (12, MAX as u32 + 1), (12, 60)] {
                let mut bad = packed.clone();
                bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                assert!(unpack(&bad).is_err());
            }
            assert!(unpack(&packed[..packed.len() - 1]).is_err());
            let mut extra = packed.clone(); extra.push(0);
            assert!(unpack(&extra).is_err());
            let mut corrupt = packed.clone(); *corrupt.last_mut().unwrap() ^= 1;
            assert!(unpack(&corrupt).is_err());
            assert_eq!(unpack(&raw).unwrap().as_ref(), raw);
        }
    }
}
