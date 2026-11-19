"""Small but structurally valid sample files for attachment tests."""

import io
import struct
import wave
import zipfile
import zlib


def png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 px, RGB
    pixels = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


def wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(b"\x00\x00" * 800)  # 0.1 s of silence
    return buffer.getvalue()


PDF = (
    b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"
)
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x64" + b"\x00" * 64
MP3_FRAME = b"\xff\xfb\x90\x64" + b"\x00" * 64  # no ID3 tag, starts with a frame
EXE = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00" + b"\x00" * 64


def office(folder: str) -> bytes:
    """A minimal OOXML package (DOCX for "word"): content types plus main part."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr(f"{folder}/document.xml", "<root/>")
    return buffer.getvalue()


def text(characters: int = 40) -> bytes:
    return ("Objednávka 12345, prosím o kontrolu. " * characters)[:characters].encode()


WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + b"\x00" * 24


def box(kind: bytes, payload: bytes) -> bytes:
    """An MP4 box: 4-byte size, 4-byte type, payload."""
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def mp4(brand: bytes, *handlers: bytes) -> bytes:
    """A minimal MP4: ftyp, mdat, then moov with one track per handler type."""
    tracks = b"".join(
        box(b"trak", box(b"mdia", box(b"hdlr", b"\x00" * 8 + handler + b"\x00" * 12)))
        for handler in handlers
    )
    return (
        box(b"ftyp", brand + b"\x00\x00\x00\x00isom")
        + box(b"mdat", b"\x00" * 32)
        + box(b"moov", tracks)
    )


M4A = mp4(b"M4A ", b"soun")
M4A_3GP = mp4(b"3gp4", b"soun")  # Android voice recorder
MP4_VIDEO = mp4(b"isom", b"vide", b"soun")
OGG_OPUS = b"OggS\x00\x02" + b"\x00" * 20 + b"\x01\x13OpusHead\x01\x01" + b"\x00" * 16
