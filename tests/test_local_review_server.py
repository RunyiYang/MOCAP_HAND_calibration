from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from local_review_server import make_handler


class LocalReviewServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.static_root = cls.base / "public"
        cls.static_root.mkdir()
        (cls.static_root / "index.html").write_bytes(b"<h1>local review</h1>")
        (cls.static_root / "media.bin").write_bytes(b"0123456789abcdef")
        (cls.base / "secret.txt").write_bytes(b"not public")

        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(cls.static_root)
        )
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address[:2]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temporary.cleanup()

    def request(
        self, method: str, target: str, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request(method, target, headers=headers or {})
            response = connection.getresponse()
            body = response.read()
            return response.status, dict(response.getheaders()), body
        finally:
            connection.close()

    def test_normal_get_and_head(self) -> None:
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"<h1>local review</h1>")
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["Cross-Origin-Opener-Policy"], "same-origin")
        self.assertIn("media-src 'self'", headers["Content-Security-Policy"])

        head_status, head_headers, head_body = self.request("HEAD", "/media.bin")
        self.assertEqual(head_status, 200)
        self.assertEqual(head_headers["Content-Length"], "16")
        self.assertEqual(head_body, b"")

    def test_single_range(self) -> None:
        status, headers, body = self.request(
            "GET", "/media.bin", {"Range": "bytes=2-5"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], "bytes 2-5/16")
        self.assertEqual(headers["Content-Length"], "4")
        self.assertEqual(body, b"2345")

    def test_suffix_range(self) -> None:
        status, headers, body = self.request(
            "GET", "/media.bin", {"Range": "bytes=-4"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], "bytes 12-15/16")
        self.assertEqual(body, b"cdef")

    def test_invalid_or_unsatisfiable_range(self) -> None:
        for value in ("bytes=99-100", "bytes=8-4", "bytes=0-1,4-5", "items=0-1"):
            with self.subTest(value=value):
                status, headers, body = self.request(
                    "GET", "/media.bin", {"Range": value}
                )
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], "bytes */16")
                self.assertEqual(headers["Accept-Ranges"], "bytes")
                self.assertEqual(body, b"")

    def test_path_traversal_is_forbidden(self) -> None:
        for target in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"):
            with self.subTest(target=target):
                status, _, body = self.request("GET", target)
                self.assertEqual(status, 403)
                self.assertNotIn(b"not public", body)


if __name__ == "__main__":
    unittest.main()
