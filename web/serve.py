"""Static server for the WebGPU app. Serves the repo root so the app (/web/),
the runtime core (/runtime/pipeline.mjs), the tokenizer (/tokenizer/llmjp_tok/),
and the ONNX artifacts (/artifacts/onnx/) are all reachable.

WebGPU needs no SharedArrayBuffer, so we skip COOP/COEP. Range requests are
supported by SimpleHTTPRequestHandler, which matters for the large .data files.

    python web/serve.py    # then open http://127.0.0.1:8137/web/
"""
from __future__ import annotations

import http.server
import socketserver
from pathlib import Path

PORT = 8137
ROOT = Path(__file__).resolve().parents[1]  # repo root


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"serving {ROOT} at http://127.0.0.1:{PORT}/")
        print(f"open: http://127.0.0.1:{PORT}/web/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
