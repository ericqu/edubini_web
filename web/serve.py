"""Serve web/ over HTTPS on the local network (camera access needs https).

    uv run python web/serve.py          # then open https://<mac-ip>:8443/ on the iPad

Uses a self-signed certificate (created on first run in web/.cert/); Safari
warns once: tap "Show Details" > "visit this website".
"""

import datetime
import http.server
import json
import os
import socket
import ssl
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.join(HERE, ".cert")
LOG_DIR = os.path.join(os.path.dirname(HERE), "results")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8443


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # no packet is sent; just picks the outgoing interface
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_cert(ip: str) -> tuple[str, str]:
    crt, key = os.path.join(CERT_DIR, "cert.pem"), os.path.join(CERT_DIR, "key.pem")
    if not (os.path.exists(crt) and os.path.exists(key)):
        os.makedirs(CERT_DIR, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                        "-keyout", key, "-out", crt, "-subj", "/CN=microscope-dev",
                        "-addext", f"subjectAltName=IP:{ip},DNS:localhost"],
                       check=True, capture_output=True)
    return crt, key


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".webmanifest": "application/manifest+json"}

    def do_POST(self):
        """/log: device reports from the app, appended to results/devices.jsonl."""
        if self.path.rstrip("/").split("/")[-1] != "log":
            self.send_error(404)
            return
        n = min(int(self.headers.get("Content-Length") or 0), 65536)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self.send_error(400)
            return
        data = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "ip": self.client_address[0], **data}
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "devices.jsonl"), "a") as fh:
            fh.write(json.dumps(data) + "\n")
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # always serve the latest build
        super().end_headers()


def main() -> None:
    ip = lan_ip()
    crt, key = ensure_cert(ip)
    os.chdir(HERE)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    sys.stdout.reconfigure(line_buffering=True)
    print(f"Serving {HERE}\n  on the iPad open:  https://{ip}:{PORT}/\n  on this Mac:       https://localhost:{PORT}/\n  camera test page:  https://{ip}:{PORT}/camtest.html\nCtrl+C to stop.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
