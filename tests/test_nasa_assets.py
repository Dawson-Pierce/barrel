import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from barrel_assets.nasa import NasaMeshConverter, SOURCES


class NasaDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.assets = Path(self.directory.name)
        self.payload = b"verified model fixture"
        self.source = dict(source="test.glb", url="https://example.invalid/model.glb",
                           sha256=hashlib.sha256(self.payload).hexdigest())
        self.source_patch = patch.dict(SOURCES, {"landsat9": self.source})
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.converter = NasaMeshConverter(self.assets)

    def test_download_saves_verified_bytes(self):
        with patch("barrel_assets.nasa.urlopen", return_value=io.BytesIO(self.payload)) as request:
            self.converter.download(("landsat9",))
        self.assertEqual((self.assets / "test.glb").read_bytes(), self.payload)
        self.assertEqual(request.call_args.args[0].full_url, self.source["url"])

    def test_valid_cache_needs_no_network(self):
        (self.assets / "test.glb").write_bytes(self.payload)
        with patch("barrel_assets.nasa.urlopen") as request:
            self.converter.download(("landsat9",))
        request.assert_not_called()

    def test_unexpected_download_is_not_saved(self):
        with patch("barrel_assets.nasa.urlopen", return_value=io.BytesIO(b"wrong model")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.converter.download(("landsat9",))
        self.assertFalse((self.assets / "test.glb").exists())

    def test_corrupt_cache_is_rejected(self):
        (self.assets / "test.glb").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.converter.download(("landsat9",))


if __name__ == "__main__":
    unittest.main()
