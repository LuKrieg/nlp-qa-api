
import json
import os
import urllib.request
import urllib.error

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_URL = os.environ.get("QA_API_BASE_URL", "").rstrip("/")
TOKEN = os.environ.get("QA_API_TOKEN", "")
HTML = Path(__file__).with_name("index.html")

if not BASE_URL.startswith("https://"):
    raise SystemExit("Configura QA_API_BASE_URL con una URL HTTPS.")

if not TOKEN:
    raise SystemExit("Configura QA_API_TOKEN en PowerShell.")


class Handler(BaseHTTPRequestHandler):

    def responder(self, status, contenido, tipo="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", tipo + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(contenido)

    def remoto(self, ruta, cuerpo=None):
        headers = {
            "Authorization": "Bearer " + TOKEN,
            "Accept": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        if cuerpo is not None:
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            BASE_URL + ruta,
            data=cuerpo,
            headers=headers,
            method="POST" if cuerpo is not None else "GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=8) as r:
                return r.status, r.read(), r.headers.get(
                    "Content-Type", "application/json"
                )
        except urllib.error.HTTPError as e:
            return e.code, e.read(), "application/json"

    def do_GET(self):
        if self.path == "/":
            self.responder(200, HTML.read_bytes(), "text/html")
        elif self.path == "/demo/health":
            self.enviar_remoto("/health")
        else:
            self.responder(404, b'{"error":"not_found"}')

    def enviar_remoto(self, ruta, cuerpo=None):
        try:
            status, data, tipo = self.remoto(ruta, cuerpo)
            self.responder(status, data, tipo.split(";")[0])
        except Exception:
            self.responder(
                502,
                b'{"error":"upstream_unavailable",'
                b'"detail":"No se pudo conectar con la API."}'
            )

    def do_POST(self):
        if self.path != "/demo/predict":
            self.responder(404, b'{"error":"not_found"}')
            return

        try:
            longitud = int(self.headers.get("Content-Length", "0"))
            if longitud <= 0 or longitud > 12000:
                raise ValueError("Tamaño de solicitud inválido")

            cuerpo = self.rfile.read(longitud)
            datos = json.loads(cuerpo)

            if not isinstance(datos, dict):
                raise ValueError("Solicitud inválida")

            if not isinstance(datos.get("context"), str) or \
               not 1 <= len(datos["context"]) <= 5000:
                raise ValueError("Contexto inválido")

            if not isinstance(datos.get("question"), str) or \
               not 1 <= len(datos["question"]) <= 500:
                raise ValueError("Pregunta inválida")

            datos = {
                "id": "demo-001",
                "context": datos["context"],
                "question": datos["question"]
            }

            self.enviar_remoto(
                "/predict", json.dumps(datos).encode("utf-8")
            )
        except (ValueError, json.JSONDecodeError) as e:
            self.responder(
                400,
                json.dumps({
                    "error": "bad_request",
                    "detail": str(e)
                }).encode("utf-8")
            )


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
