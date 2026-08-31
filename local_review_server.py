#!/usr/bin/env python3
"""Serve the built GT-calib review site locally with HTTP byte ranges.

The server is deliberately small and read-only.  It serves one resolved static
root, never lists directories, and supports the single byte ranges used by
browsers when seeking through MP4 files.
"""

from __future__ import annotations

import argparse
from email.utils import formatdate
import mimetypes
from pathlib import Path, PurePosixPath
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, Type
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = PROJECT_ROOT / "web" / "public"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8811
COPY_CHUNK_BYTES = 1024 * 1024

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "connect-src 'self'; "
    "font-src 'self' data:; "
    "frame-ancestors 'none'; "
    "img-src 'self' data:; "
    "media-src 'self'; "
    "object-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'"
)


class RangeNotSatisfiable(ValueError):
    """Raised when a Range header cannot select bytes from a resource."""


def parse_single_byte_range(value: str, size: int) -> tuple[int, int]:
    """Return an inclusive ``(start, end)`` for one RFC-style byte range.

    Multiple ranges are intentionally rejected: this local review server does
    not generate multipart responses.
    """

    match = re.fullmatch(r"bytes\s*=\s*([^,]+)", value.strip(), re.IGNORECASE)
    if match is None or size <= 0:
        raise RangeNotSatisfiable(value)

    spec = match.group(1).strip()
    bounds = re.fullmatch(r"(\d*)-(\d*)", spec)
    if bounds is None:
        raise RangeNotSatisfiable(value)

    start_text, end_text = bounds.groups()
    if not start_text and not end_text:
        raise RangeNotSatisfiable(value)

    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise RangeNotSatisfiable(value)
        start = max(0, size - suffix_length)
        return start, size - 1

    start = int(start_text)
    if start >= size:
        raise RangeNotSatisfiable(value)

    end = size - 1 if not end_text else min(int(end_text), size - 1)
    if end < start:
        raise RangeNotSatisfiable(value)
    return start, end


class StaticRangeRequestHandler(BaseHTTPRequestHandler):
    """Path-safe, read-only static file handler with single-range support."""

    static_root = DEFAULT_DIRECTORY.resolve()
    server_version = "GTCalibLocalReview/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def version_string(self) -> str:
        return self.server_version

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("X-Robots-Tag", "noindex, noarchive, nosnippet")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _resolve_request_path(self) -> Path | None:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc:
            return None

        try:
            decoded = unquote(parsed.path, errors="strict")
        except UnicodeError:
            return None
        if "\x00" in decoded or "\\" in decoded:
            return None

        relative = PurePosixPath(decoded.lstrip("/"))
        if any(part == ".." for part in relative.parts):
            return None

        candidate = self.static_root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.static_root)
        except (OSError, ValueError):
            return None

        if resolved.is_dir():
            resolved = (resolved / "index.html").resolve(strict=False)
            try:
                resolved.relative_to(self.static_root)
            except ValueError:
                return None
        return resolved

    def _serve(self, *, send_body: bool) -> None:
        path = self._resolve_request_path()
        if path is None:
            self.send_error(403, "Forbidden")
            return
        if not path.is_file():
            self.send_error(404, "File not found")
            return

        try:
            size = path.stat().st_size
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return

        with stream:
            range_header = self.headers.get("Range")
            if range_header is None:
                start, end, status = 0, size - 1, 200
            else:
                try:
                    start, end = parse_single_byte_range(range_header, size)
                except RangeNotSatisfiable:
                    self.send_response(416)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206

            length = end - start + 1 if size else 0
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header(
                "Last-Modified", formatdate(path.stat().st_mtime, usegmt=True)
            )
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            if not send_body or length == 0:
                return
            stream.seek(start)
            self._copy_bytes(stream, length)

    def _copy_bytes(self, stream: BinaryIO, remaining: int) -> None:
        try:
            while remaining:
                chunk = stream.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Seeking browsers routinely cancel an obsolete video range.
            return

    def log_message(self, format: str, *args: object) -> None:
        # Keep the default useful request log, but include the bound endpoint.
        super().log_message(format, *args)


def make_handler(directory: Path) -> Type[StaticRangeRequestHandler]:
    """Create a handler class pinned to one resolved static directory."""

    root = directory.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    class BoundStaticRangeRequestHandler(StaticRangeRequestHandler):
        static_root = root

    return BoundStaticRangeRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve web/public locally with safe HTTP byte-range support."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handler = make_handler(args.directory)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    host, port = server.server_address[:2]
    print(f"Serving {handler.static_root} at http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local review server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
