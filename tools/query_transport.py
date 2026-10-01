"""Decode the bounded, lossless LAYZ query envelope; retain wire bytes for sending."""
import struct
import zlib

from measure_inference import Failure

MAX_STATE_BYTES = 60 + 128 * 4 + 7 * 4 + 128 * 2048 * 4
AUTO_BUNDLE = "bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092"


def inspect_state(wire):
    """Validate the complete wire stream but retain only the unshuffled header.

    Discard tensor chunks after checksum/length validation. The original wire
    remains the continuation; no float planes need reconstructing on the client.
    """
    if len(wire) > MAX_STATE_BYTES:
        raise Failure("query state exceeds transport limit")
    codec = None
    if wire.startswith(b"LAYZ"):
        if len(wire) < 16:
            raise Failure("truncated query transport header")
        version, codec, size = struct.unpack_from("<III", wire, 4)
        if version != 1 or codec not in (1, 2) or not 60 <= size <= MAX_STATE_BYTES:
            raise Failure("unsupported query transport header")
        decoder = zlib.decompressobj()
        pending = wire[16:]
        prefix = bytearray()
        count = 0
        try:
            while pending:
                chunk = decoder.decompress(pending, min(65536, size + 1 - count))
                count += len(chunk)
                prefix.extend(chunk[:max(0, 60 - len(prefix))])
                if count > size:
                    raise Failure("compressed query state length differs")
                pending = decoder.unconsumed_tail
                if decoder.eof:
                    break
        except zlib.error as error:
            raise Failure("invalid compressed query state") from error
        if count != size or not decoder.eof or decoder.unused_data or pending:
            raise Failure("compressed query state length differs")
        header = bytes(prefix)
    else:
        size, header = len(wire), wire[:60]
    if len(header) < 60 or header[:4] not in (b"LAYQ", b"LAYI"):
        raise Failure("invalid continuation header")
    if codec is not None and header[:4] != (b"LAYI" if codec == 1 else b"LAYQ"):
        raise Failure("query transport codec differs from state")
    tokens, markers = struct.unpack_from("<II", header, 52)
    offset = 60 + 4 * (tokens + markers)
    if not 1 <= tokens <= 128 or not 2 <= markers <= 7 or offset > size:
        raise Failure("invalid continuation lengths")
    if header[:4] == b"LAYQ" and (size - offset) % 4:
        raise Failure("invalid shuffled query state")
    return header, size


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
    if tokens <= 16:
        width = 16
    elif tokens <= 43:
        width = 9 if mode == "f32" else 8
    elif tokens <= 65:
        width = 6
    elif tokens <= 72:
        width = 6 if mode == "f32" else 5
    elif tokens <= (96 if mode == "f32" else 90):
        width = 4
    else:
        width = 3
    width = min(width, cap)
    encoder_left = max(0, 28 - completed)
    if encoder_left == 0 or encoder_left == 1 or encoder_left + 2 <= width:
        return min(16, cap, 32 - completed)
    return min(width, 32 - completed)
