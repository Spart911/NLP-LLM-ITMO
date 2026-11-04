"""Локальный интерфейс выбора прогона."""

import contextlib
import io
import json
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from catalog.ask import answer
from catalog.config import redact
from catalog.pipeline import build_pipeline
from catalog.registry import BY_ID, FLOWS

HOST = "127.0.0.1"
PORT = 8765
PAGE = Path(__file__).resolve().parent / "static" / "index.html"
RESULTS = Path(__file__).resolve().parents[1] / "results"
REPORT = Path(__file__).resolve().parents[1] / "report.md"

_runs: dict[str, dict] = {}
_lock = threading.Lock()
_busy = False


def serve(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Интерфейс: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._bytes(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/flows":
            public = [
                {
                    "id": "ask",
                    "title": "Запрос",
                    "summary": "Вопрос проходит итоговый пайплайн: промпт, модель, разбор с повтором, журнал и поиск.",
                    "kind": "ask",
                }
            ]
            public.extend(
                {"id": item["id"], "title": item["title"], "summary": item["summary"], "kind": "run"}
                for item in FLOWS
            )
            public.append(
                {
                    "id": "report",
                    "title": "Отчет",
                    "summary": "Цель, архитектура, результаты прогонов, сравнения и ответы на контрольные вопросы.",
                    "kind": "document",
                }
            )
            self._json(200, public)
            return
        if path == "/api/report":
            if not REPORT.is_file():
                self._json(404, {"error": "Отчет не найден"})
                return
            text = redact(REPORT.read_text(encoding="utf-8"))
            self._json(200, {"markdown": text})
            return
        if path.startswith("/api/runs/"):
            run_id = path.removeprefix("/api/runs/")
            with _lock:
                stored = _runs.get(run_id)
                run = dict(stored) if stored is not None else None
            if run is None:
                self._json(404, {"error": "Прогон не найден"})
                return
            self._json(200, run)
            return
        if path.startswith("/api/results/"):
            flow_id = path.removeprefix("/api/results/")
            saved = _saved(flow_id)
            if saved is None:
                self._json(404, {"error": "Сохраненного прогона нет"})
                return
            self._json(200, saved)
            return
        self._json(404, {"error": "Не найдено"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Нужен JSON"})
            return
        if path == "/api/ask":
            query = str(payload.get("query") or "").strip()
            if not query:
                self._json(400, {"error": "Нужен вопрос"})
                return
            if len(query) > 1000:
                self._json(400, {"error": "Слишком длинный вопрос"})
                return
            self._start("ask", lambda: _ask(query))
            return
        if path != "/api/runs":
            self._json(404, {"error": "Не найдено"})
            return
        flow_id = payload.get("flow")
        if flow_id not in BY_ID:
            self._json(400, {"error": "Неизвестный сценарий"})
            return
        self._start(flow_id, BY_ID[flow_id]["run"])

    def _start(self, flow_id: str, worker) -> None:
        global _busy
        with _lock:
            if _busy:
                self._json(409, {"error": "Уже идет другой прогон"})
                return
            _busy = True
            run_id = uuid.uuid4().hex[:12]
            _runs[run_id] = {
                "id": run_id,
                "flow": flow_id,
                "status": "running",
                "output": "",
                "pipeline": None,
                "report": None,
            }
        threading.Thread(target=_execute, args=(run_id, flow_id, worker), daemon=True).start()
        self._json(202, {"id": run_id})

    def log_message(self, format: str, *args) -> None:
        return

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self._bytes(status, body, "application/json; charset=utf-8")

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _LiveLog(io.TextIOBase):
    def __init__(self, run_id: str) -> None:
        self._run_id = run_id

    def write(self, text: str) -> int:
        if text:
            with _lock:
                merged = _runs[self._run_id]["output"] + text
                _runs[self._run_id]["output"] = redact(merged)
        return len(text)

    def flush(self) -> None:
        return


def _ask(query: str) -> dict:
    print(f"Вопрос: {query}")
    result = answer(query)
    if result["error"]:
        print(result["error"])
        return result
    print(result["params"])
    for line in result["products"]:
        print(f"- {line}")
    if result["message"]:
        print(result["message"])
    return result


def _execute(run_id: str, flow_id: str, worker) -> None:
    global _busy
    log = _LiveLog(run_id)
    status = "done"
    result = None
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if flow_id != "ask":
                print(f"{BY_ID[flow_id]['title']}: прогон запущен.")
            result = worker()
    except Exception:
        status = "error"
        log.write(redact(traceback.format_exc()))
    report = None
    pipeline = None
    if status == "done" and isinstance(result, dict):
        clean = json.loads(redact(json.dumps(result, ensure_ascii=False)))
        pipeline = clean.pop("pipeline", None)
        report = clean
        if pipeline is None:
            pipeline = build_pipeline(flow_id, report)
    with _lock:
        _runs[run_id]["status"] = status
        _runs[run_id]["report"] = report
        _runs[run_id]["pipeline"] = pipeline
        _busy = False


def _saved(flow_id: str) -> dict | None:
    if flow_id not in BY_ID:
        return None
    path = RESULTS / f"{flow_id}.json"
    if not path.is_file():
        return None
    report = json.loads(redact(path.read_text(encoding="utf-8")))
    return {
        "flow": flow_id,
        "status": "saved",
        "output": "",
        "report": report,
        "pipeline": build_pipeline(flow_id, report),
    }
