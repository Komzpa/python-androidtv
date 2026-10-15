import base64
import gzip
import struct
import sys
import unittest
import zlib
from unittest.mock import Mock, patch


sys.path.insert(0, "..")

from androidtv import framebuffer
from androidtv.basetv.basetv_sync import BaseTVSync


PNG_IMAGE = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x06\x00\x00\x00\x8d2\xcf\xbd\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\x00\x00\tpHYs\x00\x00\x0fa\x00\x00\x0fa\x01\xa8?\xa7i\x00\x00\x00\x0eIDAT\x18\x95c`\x18\x05\x83\x13\x00\x00\x01\x9a\x00\x01\x16\xca\xd3i\x00\x00\x00\x00IEND\xaeB`\x82"


def _header(width, height, pixel_format, dataspace=None):
    if dataspace is None:
        return struct.pack("<III", width, height, pixel_format)
    return struct.pack("<IIII", width, height, pixel_format, dataspace)


def _parse_png(png):
    """Return (width, height, color_type, scanlines) for a PNG written by framebuffer."""
    assert png.startswith(framebuffer.PNG_SIGNATURE)
    offset = len(framebuffer.PNG_SIGNATURE)
    chunks = {}
    while offset < len(png):
        (length,) = struct.unpack_from(">I", png, offset)
        tag = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack_from(">I", png, offset + 8 + length)
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF
        chunks[tag] = data
        offset += 12 + length
    width, height, depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert depth == 8
    return width, height, color_type, zlib.decompress(chunks[b"IDAT"])


def _rows(scanlines, height, stride):
    """Strip the per-row filter byte and return the pixel bytes of each row."""
    row_len = stride + 1
    rows = []
    for row in range(height):
        line = scanlines[row * row_len : (row + 1) * row_len]
        assert line[0] == 0
        rows.append(line[1:])
    return rows


class TestDecodeFramebuffer(unittest.TestCase):
    def test_rgba_header_12(self):
        raw = _header(2, 1, 1) + bytes([1, 2, 3, 4, 5, 6, 7, 8])
        width, height, color_type, scanlines = _parse_png(framebuffer.decode_framebuffer(raw))
        self.assertEqual((width, height, color_type), (2, 1, 6))
        self.assertEqual(_rows(scanlines, 1, 8), [bytes([1, 2, 3, 4, 5, 6, 7, 8])])

    def test_rgba_header_16_with_dataspace(self):
        raw = _header(1, 2, 1, dataspace=0) + bytes(range(8))
        width, height, color_type, scanlines = _parse_png(framebuffer.decode_framebuffer(raw))
        self.assertEqual((width, height, color_type), (1, 2, 6))
        self.assertEqual(_rows(scanlines, 2, 4), [bytes(range(4)), bytes(range(4, 8))])

    def test_bgra_is_swapped_to_rgba(self):
        raw = _header(1, 1, 5) + bytes([10, 20, 30, 40])
        _, _, color_type, scanlines = _parse_png(framebuffer.decode_framebuffer(raw))
        self.assertEqual(color_type, 6)
        self.assertEqual(_rows(scanlines, 1, 4), [bytes([30, 20, 10, 40])])

    def test_rgbx_drops_padding_byte(self):
        raw = _header(2, 1, 2) + bytes([1, 2, 3, 0, 4, 5, 6, 0])
        _, _, color_type, scanlines = _parse_png(framebuffer.decode_framebuffer(raw))
        self.assertEqual(color_type, 2)
        self.assertEqual(_rows(scanlines, 1, 6), [bytes([1, 2, 3, 4, 5, 6])])

    def test_rgb888(self):
        raw = _header(1, 2, 3) + bytes([1, 2, 3, 4, 5, 6])
        _, _, color_type, scanlines = _parse_png(framebuffer.decode_framebuffer(raw))
        self.assertEqual(color_type, 2)
        self.assertEqual(_rows(scanlines, 2, 3), [bytes([1, 2, 3]), bytes([4, 5, 6])])

    def test_rgb565_expands_to_8_bit(self):
        white = struct.pack("<H", 0xFFFF)
        red = struct.pack("<H", 0xF800)
        raw = _header(2, 1, 4) + white + red
        _, _, color_type, scanlines = _parse_png(framebuffer.decode_framebuffer(raw))
        self.assertEqual(color_type, 2)
        self.assertEqual(_rows(scanlines, 1, 6), [bytes([255, 255, 255, 255, 0, 0])])

    def test_unsupported_format_raises(self):
        raw = _header(1, 1, 43) + bytes(4)
        with self.assertRaises(framebuffer.UnsupportedScreencapFormatError):
            framebuffer.decode_framebuffer(raw)

    def test_truncated_header_raises(self):
        with self.assertRaises(framebuffer.ScreencapError):
            framebuffer.decode_framebuffer(bytes(11))

    def test_truncated_pixels_raise(self):
        raw = _header(2, 2, 1) + bytes(4)
        with self.assertRaises(framebuffer.ScreencapError):
            framebuffer.decode_framebuffer(raw)

    def test_zero_dimensions_raise(self):
        with self.assertRaises(framebuffer.ScreencapError):
            framebuffer.decode_framebuffer(_header(0, 1, 1) + bytes(4))


class TestDecodeCompressedScreencap(unittest.TestCase):
    def _encode(self, raw):
        return base64.b64encode(gzip.compress(raw)).decode()

    def test_round_trip(self):
        raw = _header(1, 1, 1) + bytes([9, 8, 7, 6])
        png = framebuffer.decode_compressed_screencap(self._encode(raw))
        self.assertEqual(_parse_png(png)[3], b"\x00" + bytes([9, 8, 7, 6]))

    def test_corrupt_gzip_raises_screencap_error(self):
        encoded = base64.b64encode(gzip.compress(_header(1, 1, 1) + bytes(4))[:-6]).decode()
        with self.assertRaises(framebuffer.ScreencapError):
            framebuffer.decode_compressed_screencap(encoded)

    def test_invalid_base64_raises_screencap_error(self):
        with self.assertRaises(framebuffer.ScreencapError):
            framebuffer.decode_compressed_screencap("not base64!")

    def test_truncated_compressed_frame_raises(self):
        with self.assertRaises(framebuffer.ScreencapError):
            framebuffer.decode_compressed_screencap(self._encode(bytes(11)))


class TestAdbScreencap(unittest.TestCase):
    def setUp(self):
        self.btv = BaseTVSync.__new__(BaseTVSync)
        self.btv._adb = Mock()

    def test_returns_none_when_shell_not_text(self):
        with patch.object(self.btv, "adb_shell", return_value=None):
            self.assertIsNone(self.btv.adb_screencap())

    def test_falls_back_when_device_has_no_compression(self):
        self.btv._adb.screencap.return_value = PNG_IMAGE
        with patch.object(self.btv, "adb_shell", return_value=""):
            self.assertEqual(self.btv.adb_screencap(), PNG_IMAGE)
            self.btv._adb.screencap.assert_called_once_with()

    def test_falls_back_for_unsupported_format(self):
        encoded = base64.b64encode(gzip.compress(_header(1, 1, 43) + bytes(4))).decode()
        self.btv._adb.screencap.return_value = PNG_IMAGE
        with patch.object(self.btv, "adb_shell", return_value=encoded):
            self.assertEqual(self.btv.adb_screencap(), PNG_IMAGE)

    def test_decodes_compressed_frame(self):
        raw = _header(1, 1, 1) + bytes([1, 2, 3, 4])
        encoded = base64.b64encode(gzip.compress(raw)).decode()
        with patch.object(self.btv, "adb_shell", return_value=encoded) as shell:
            png = self.btv.adb_screencap()
        shell.assert_called_once_with(framebuffer.COMPRESSED_SCREENCAP_COMMAND)
        self.assertEqual(_parse_png(png)[0:2], (1, 1))


if __name__ == "__main__":
    unittest.main()
