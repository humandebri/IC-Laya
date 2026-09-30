"""Decode the bounded, lossless LAYZ query envelope; retain wire bytes for sending."""
import struct
import zlib

from measure_inference import Failure

MAX_STATE_BYTES = 60 + 128 * 4 + 7 * 4 + 128 * 2048 * 4
AUTO_BUNDLE = "bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092"


def unpack_state(wire):
    if len(wire) > MAX_STATE_BYTES:
        raise Failure("query state exceeds transport limit")
    if not wire.startswith(b"LAYZ"):
        return wire
    if len(wire) < 16:
        raise Failure("truncated query transport header")
    version, codec, size = struct.unpack_from("<III", wire, 4)
    if version != 1 or codec not in (1, 2) or not 60 <= size <= MAX_STATE_BYTES:
        raise Failure("unsupported query transport header")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(wire[16:], size + 1)
    except zlib.error as error:
        raise Failure("invalid compressed query state") from error
    if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise Failure("compressed query state length differs")
    if codec == 1:
        if not raw.startswith(b"LAYI"):
            raise Failure("INT8 transport codec requires INT8 state")
        return raw
    if len(raw) < 60 or raw[:4] != b"LAYQ":
        raise Failure("F32 transport codec requires F32 state")
    tokens, markers = struct.unpack_from("<II", raw, 52)
    offset = 60 + 4 * (tokens + markers)
    if not 1 <= tokens <= 128 or not 2 <= markers <= 7 or offset > len(raw) or (len(raw)-offset) % 4:
        raise Failure("invalid shuffled query state")
    planes = raw[offset:]
    n = len(planes) // 4
    restored = bytearray(len(planes))
    for lane in range(4):
        restored[lane::4] = planes[lane*n:(lane+1)*n]
    return raw[:offset] + restored


def auto_width(tokens, mode, completed, cap=16):
    """Empirical fixed-pack tiers, with conservative head merging and fallback.

    A tier is a starting choice, not a worst-case instruction guarantee. The
    caller halves the cap on an IC instruction-limit rejection.
    """
    width = 16 if tokens <= 16 else (9 if mode == "f32" else 8) if tokens <= 43 else 6 if tokens <= 65 else 3
    width = min(width, cap)
    encoder_left = max(0, 28 - completed)
    if encoder_left == 0 or encoder_left == 1 or encoder_left + 2 <= width:
        return min(16, cap, 32 - completed)
    return min(width, 32 - completed)
