"""16-bit greyscale PNG, with nothing but the standard library.

##### THIS EXISTS SO METRIC DEPTH COSTS THE HOST NO NEW DEPENDENCY. #####

`local_planner` measures distances out of a depth image the robot sends as a
16-bit PNG -- the one lossless encoding for depth that every image library on
the robot side already writes (OpenCV in the vision node, Pillow in the
simulator). Reading it back could have pulled Pillow onto the mission host,
which runs on an operator's laptop and whose whole install today is
`paho-mqtt` and `cbor2`. Sixty lines of zlib is a better trade than a new
binary wheel in the boot path of the thing that drives a robot.

⚠️ 16-BIT GREYSCALE, NON-INTERLACED, AND NOTHING ELSE. This is not a PNG
library. It decodes the one shape depth arrives in and returns None for
everything else, so a colour PNG or an interlaced one is a clean refusal rather
than a plausible array of wrong numbers. All five PNG row filters are handled,
because the encoder on the other end picks them and none of them is optional.

Samples are big-endian, as PNG requires -- the classic silent bug here is
reading them native-endian on a little-endian host, which turns 1234 mm into
53764 mm and looks like a sensor fault.
"""
from __future__ import annotations

import struct
import zlib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Bytes per sample-group for 16-bit greyscale. The filters below are defined in
#: terms of "the pixel to the left", which for this format is two bytes back.
_BPP = 2


def encode_png16(values, width: int, height: int) -> bytes:
    """A 16-bit greyscale PNG from row-major unsigned samples.

    Filter 0 (None) on every row: depth is not a photograph and the adaptive
    filters that help continuous-tone images mostly do not help a depth map with
    large flat runs, which zlib already compresses well.
    """
    if width <= 0 or height <= 0:
        raise ValueError("png size must be positive")
    if len(values) != width * height:
        raise ValueError(f"expected {width * height} samples, got {len(values)}")
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter: None
        start = row * width
        raw.extend(struct.pack(f">{width}H", *values[start:start + width]))
    return b"".join(
        (
            PNG_MAGIC,
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(bytes(raw), 6)),
            _chunk(b"IEND", b""),
        )
    )


def decode_png16(data: bytes) -> tuple[list[int], int, int] | None:
    """(samples, width, height) from a 16-bit greyscale PNG, or None.

    None rather than an exception for every malformed input: this decodes frames
    off a radio link several times a second, and one bad frame must cost a
    ranging fallback, not a mission.
    """
    try:
        return _decode(data)
    except Exception:  # noqa: BLE001 - see docstring
        return None


def _decode(data: bytes) -> tuple[list[int], int, int] | None:
    if not data.startswith(PNG_MAGIC):
        return None
    offset = len(PNG_MAGIC)
    header: tuple[int, ...] | None = None
    payload = bytearray()
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        offset += 12 + length  # length + type + body + CRC
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            # Several IDAT chunks are one zlib stream split up, not several
            # streams: concatenate first, inflate once.
            payload.extend(body)
        elif kind == b"IEND":
            break
    if header is None or not payload:
        return None
    width, height, depth, colour, compression, filter_method, interlace = header
    if (depth, colour, compression, filter_method, interlace) != (16, 0, 0, 0, 0):
        return None
    if width <= 0 or height <= 0:
        return None

    raw = zlib.decompress(bytes(payload))
    stride = width * _BPP
    if len(raw) != (stride + 1) * height:
        return None

    out: list[int] = []
    previous = bytearray(stride)
    position = 0
    for _ in range(height):
        filter_type = raw[position]
        line = bytearray(raw[position + 1:position + 1 + stride])
        position += 1 + stride
        _unfilter(filter_type, line, previous, stride)
        out.extend(struct.unpack(f">{width}H", bytes(line)))
        previous = line
    return (out, width, height)


def _unfilter(filter_type: int, line: bytearray, previous: bytearray, stride: int) -> None:
    """Undo one PNG row filter in place. Filters are defined on BYTES, not samples."""
    if filter_type == 0:
        return
    for index in range(stride):
        left = line[index - _BPP] if index >= _BPP else 0
        up = previous[index]
        if filter_type == 1:
            addend = left
        elif filter_type == 2:
            addend = up
        elif filter_type == 3:
            addend = (left + up) // 2
        elif filter_type == 4:
            addend = _paeth(left, up, previous[index - _BPP] if index >= _BPP else 0)
        else:
            raise ValueError(f"unknown png filter {filter_type}")
        line[index] = (line[index] + addend) & 0xFF


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _chunk(kind: bytes, body: bytes) -> bytes:
    return b"".join(
        (struct.pack(">I", len(body)), kind, body, struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))
    )


__all__ = ["decode_png16", "encode_png16"]
