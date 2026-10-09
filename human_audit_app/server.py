#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
SAMPLES_PATH = APP_DIR / "data" / "audit_samples_50.csv"
ANNOTATIONS_DIR = REPO_DIR / "human_annotations"


OUTPUT_COLUMNS = [
    "audit_id",
    "dataset",
    "example_id",
    "A_human_sim",
    "decision",
    "reason_category",
    "annotation_note",
    "simulated_annotator",
    "provenance",
]

EDIT_REASONS = {
    "",
    "Factual or reasoning error",
    "Incomplete",
    "Does not follow the request",
    "Clarity or presentation issue",
    "Other",
}

WRITE_LOCK = threading.RLock()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def load_samples() -> list[dict[str, str]]:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Audit data not found: {SAMPLES_PATH}")
    with SAMPLES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [{k: _clean_text(v) for k, v in row.items()} for row in csv.DictReader(f)]

    if len(rows) != 200:
        raise RuntimeError(f"Expected exactly 200 audit samples, found {len(rows)}")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    expected = {"GPQA": 50, "MATH-500": 50, "MMLU-Pro": 50, "PopQA": 50}
    if counts != expected:
        raise RuntimeError(f"Unexpected dataset composition: {counts}; expected {expected}")
    return rows


SAMPLES = load_samples()
SAMPLE_BY_ID = {row["audit_id"]: row for row in SAMPLES}


def safe_annotator_name(raw: str) -> str:

    name = re.sub(r"\s+", " ", _clean_text(raw)).strip()
    if not name:
        raise ValueError("Please enter your name.")
    if len(name) > 80:
        raise ValueError("Name must be 80 characters or fewer.")

    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    if not name or name in {".", ".."}:
        raise ValueError("Please enter a valid name.")
    return name


def annotation_path(name: str) -> Path:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return ANNOTATIONS_DIR / f"{safe_annotator_name(name)}.csv"


def read_annotations(name: str) -> dict[str, dict[str, str]]:
    path = annotation_path(name)
    if not path.exists():

        atomic_write_annotations(path, {})
        return {}

    rows: dict[str, dict[str, str]] = {}
    with WRITE_LOCK, path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            audit_id = _clean_text(row.get("audit_id"))
            if audit_id in SAMPLE_BY_ID:
                rows[audit_id] = {col: _clean_text(row.get(col, "")) for col in OUTPUT_COLUMNS}
    return rows


def atomic_write_annotations(path: Path, rows: dict[str, dict[str, str]]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with WRITE_LOCK:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for sample in SAMPLES:
                record = rows.get(sample["audit_id"])
                if record:
                    writer.writerow({col: record.get(col, "") for col in OUTPUT_COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def first_unreviewed_index(annotations: dict[str, dict[str, str]]) -> int:
    for i, sample in enumerate(SAMPLES):
        if sample["audit_id"] not in annotations:
            return i
    return max(0, len(SAMPLES) - 1)


def public_annotation(row: dict[str, str]) -> dict[str, Any]:
    try:
        a_value: int | None = int(row["A_human_sim"])
    except (ValueError, TypeError, KeyError):
        a_value = None
    return {
        "decision": row.get("decision", ""),
        "A": a_value,
        "reason_category": row.get("reason_category", ""),
        "annotation_note": row.get("annotation_note", ""),
    }


def session_payload(name: str) -> dict[str, Any]:
    safe_name = safe_annotator_name(name)
    annotations = read_annotations(safe_name)
    public_samples = [
        {
            "audit_id": s["audit_id"],
            "dataset": s["dataset"],
            "example_id": s["example_id"],
            "question": s["question"],
            "gold_answer": s["gold_answer"],
            "gold_final": s["gold_final"],
            "model_answer": s["model_answer"],
        }
        for s in SAMPLES
    ]
    return {
        "name": safe_name,
        "filename": f"{safe_name}.csv",
        "samples": public_samples,
        "annotations": {k: public_annotation(v) for k, v in annotations.items()},
        "start_index": first_unreviewed_index(annotations),
        "completed": len(annotations),
        "total": len(SAMPLES),
    }


def save_annotation(payload: dict[str, Any]) -> dict[str, Any]:
    name = safe_annotator_name(str(payload.get("name", "")))
    audit_id = _clean_text(payload.get("audit_id"))
    if audit_id not in SAMPLE_BY_ID:
        raise ValueError("Unknown audit sample.")

    decision_raw = payload.get("decision")
    decision = "" if decision_raw is None else _clean_text(decision_raw)
    if decision not in {"", "Confirm", "Edit"}:
        raise ValueError("Decision must be Confirm, Edit, or empty.")

    path = annotation_path(name)
    with WRITE_LOCK:
        rows = read_annotations(name)
        if decision == "":
            rows.pop(audit_id, None)
        else:
            sample = SAMPLE_BY_ID[audit_id]
            reason = _clean_text(payload.get("reason_category", "")).strip()
            note = _clean_text(payload.get("annotation_note", "")).strip()
            if len(note) > 5000:
                note = note[:5000]

            if decision == "Confirm":
                a_value = "0"
                reason = "Correct as-is"
            else:
                a_value = "1"
                if reason not in EDIT_REASONS:
                    raise ValueError("Unknown edit-reason category.")

            rows[audit_id] = {
                "audit_id": audit_id,
                "dataset": sample["dataset"],
                "example_id": sample["example_id"],
                "A_human_sim": a_value,
                "decision": decision,
                "reason_category": reason,
                "annotation_note": note,
                "simulated_annotator": name,
                "provenance": "Human annotation via Human Response Audit UI",
            }
        atomic_write_annotations(path, rows)

    return {
        "ok": True,
        "filename": path.name,
        "completed": len(rows),
        "annotation": public_annotation(rows[audit_id]) if audit_id in rows else None,
    }


class AuditHandler(BaseHTTPRequestHandler):
    server_version = "HumanAudit/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:

        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _serve_file(self, path: Path) -> None:
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if STATIC_DIR.resolve() not in path.parents and path != (STATIC_DIR / "index.html").resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        content = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (mime or "application/octet-stream") + ("; charset=utf-8" if (mime or "").startswith("text/") else ""))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/session":
            try:
                params = urllib.parse.parse_qs(parsed.query)
                name = params.get("name", [""])[0]
                self._json({"ok": True, **session_payload(name)})
            except ValueError as exc:
                self._error(str(exc))
            return

        if parsed.path == "/api/export":
            try:
                params = urllib.parse.parse_qs(parsed.query)
                name = safe_annotator_name(params.get("name", [""])[0])
                path = annotation_path(name)
                if not path.exists():
                    atomic_write_annotations(path, {})
                data = path.read_bytes()
                encoded = urllib.parse.quote(path.name)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded}")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except ValueError as exc:
                self._error(str(exc))
            return

        if parsed.path in {"/", "/index.html"}:
            self._serve_file(STATIC_DIR / "index.html")
            return

        if parsed.path.startswith("/static/"):
            relative = urllib.parse.unquote(parsed.path[len("/static/"):])
            candidate = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in candidate.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._serve_file(candidate)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 50_000:
                raise ValueError("Request is too large.")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Invalid request payload.")
            self._json(save_annotation(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(str(exc))
        except Exception as exc:
            self._error(f"Could not save annotation: {exc}", status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Human Response Audit UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    local_url = f"http://127.0.0.1:{args.port}"
    print("\nHuman Response Audit")
    print(f"  App:         {local_url}")
    print(f"  Samples:     {len(SAMPLES)} (50 × 4 datasets)")
    print(f"  Annotations: {ANNOTATIONS_DIR}")
    print("  Stop:        Ctrl+C\n")

    if not args.no_browser and args.host in {"127.0.0.1", "localhost"}:
        threading.Timer(0.7, lambda: webbrowser.open(local_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping audit server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
