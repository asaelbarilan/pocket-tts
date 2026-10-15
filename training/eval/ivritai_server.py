"""Local checkpoint evaluation dashboard.

Run from the repository environment with::

    python -m training.eval.ivritai_server

Then open http://127.0.0.1:8001. The evaluator runs in a separate process using
the same Python environment; the dashboard itself does not load Torch/models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path must be inside the selected folder")
    return path


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class SentenceSelection(BaseModel):
    file: str
    line: int = Field(ge=1)


class EvalSettings(BaseModel):
    device: str = "cuda"
    asr: str = ""
    asr_compute_type: str = "int8_float16"
    beam_size: int = Field(default=5, ge=1)
    batch_size: int = Field(default=6, ge=1)
    temp: float = Field(default=0.3, ge=0, allow_inf_nan=False)
    cfg: float = Field(default=2.0, allow_inf_nan=False)
    n_steps: int = Field(default=1, ge=1)
    eos_threshold: float = Field(default=-1.0, allow_inf_nan=False)
    max_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    use_ema: bool = True
    bootstrap_samples: int = Field(default=1000, ge=1)


class EvaluationRequest(BaseModel):
    run_dir: str
    checkpoints: list[str] = Field(min_length=1)
    voices: list[str] = Field(min_length=1)
    sentences: list[SentenceSelection] = Field(default_factory=list)
    custom_sentences: str = ""
    label: str = Field(default="", max_length=80)
    devices: list[str] = Field(default_factory=list)
    settings: EvalSettings = Field(default_factory=EvalSettings)


class EvaluationManager:
    def __init__(self, runs_dir: Path, eval_data: Path, output_dir: Path):
        self.runs_dir = runs_dir.expanduser().resolve()
        self.eval_data = eval_data.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.lock = threading.Lock()
        self.active: str | None = None
        self.process: subprocess.Popen | None = None
        self.worker: threading.Thread | None = None
        self.cancelled = False
        self.closing = False
        # A previous server cannot still supervise these jobs. Keep their artifacts.
        if self.output_dir.exists():
            for path in self.output_dir.glob("*/manifest.json"):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") == "running":
                    manifest.update(status="interrupted", finished_at=now())
                    write_json(path, manifest)

    def run_path(self, name: str) -> Path:
        path = Path(name).expanduser()
        path = (path if path.is_absolute() else self.runs_dir / path).resolve()
        if not (path / "args.yaml").is_file():
            raise ValueError(f"missing training metadata: {path / 'args.yaml'}")
        return path

    def checkpoints(self, run_dir: str) -> list[dict]:
        run = self.run_path(run_dir)
        return [
            {"name": path.name, **fingerprint(path)}
            for path in sorted(run.glob("checkpoint_*.pt"))
            if path.is_file() and re.fullmatch(r"checkpoint_\d+\.pt", path.name)
        ]

    def catalog(self) -> dict:
        runs = []
        if (self.runs_dir / "args.yaml").is_file():
            runs.append({"name": self.runs_dir.name, "path": str(self.runs_dir)})
        if self.runs_dir.exists():
            for args in sorted(self.runs_dir.rglob("args.yaml")):
                run = args.parent.resolve()
                if run == self.runs_dir or run.is_relative_to(self.output_dir):
                    continue
                runs.append({"name": str(run.relative_to(self.runs_dir)), "path": str(run)})
        voices, sentences = [], []
        if self.eval_data.exists():
            for path in sorted(self.eval_data.rglob("*")):
                if not path.is_file():
                    continue
                name = path.relative_to(self.eval_data).as_posix()
                if path.suffix.lower() in AUDIO_EXTENSIONS:
                    voices.append({"name": name, "size_bytes": path.stat().st_size})
                elif path.suffix.lower() == ".txt":
                    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if text.strip():
                            sentences.append({"file": name, "line": line, "text": text.strip()})
        return {
            "runs_dir": str(self.runs_dir),
            "eval_data": str(self.eval_data),
            "output_dir": str(self.output_dir),
            "runs": runs,
            "voices": voices,
            "sentences": sentences,
            "defaults": EvalSettings().model_dump(),
        }

    def folder(self, evaluation_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", evaluation_id):
            raise ValueError("invalid evaluation ID")
        folder = inside(self.output_dir, evaluation_id)
        if not any((folder / name).is_file() for name in ("manifest.json", "evaluation.json")):
            raise FileNotFoundError("evaluation not found")
        return folder

    def manifest(self, folder: Path) -> dict:
        path = folder / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        # Standalone CLI evaluations are viewable without dashboard-specific files.
        evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        voices = [{"name": Path(path).name, "path": path} for path in evaluation["voices"]]
        return {
            "id": folder.name,
            "label": "CLI evaluation",
            "managed": False,
            "status": evaluation["status"],
            "created_at": evaluation["started_at"],
            "started_at": evaluation["started_at"],
            "finished_at": evaluation["finished_at"],
            "run_dir": evaluation["run_dir"],
            "output_dir": str(folder),
            "checkpoints": [{"path": job["checkpoint"]} for job in evaluation["jobs"]],
            "voices": voices,
            "sentences": [{"text": text} for text in evaluation["sentences"]],
            "settings": evaluation["settings"],
            "devices": evaluation["devices"],
            "expected_audio_count": len(evaluation["jobs"])
            * len(voices)
            * len(evaluation["sentences"]),
        }

    def history(self) -> list[dict]:
        if not self.output_dir.exists():
            return []
        folders = {
            path.parent
            for name in ("manifest.json", "evaluation.json")
            for path in self.output_dir.glob(f"*/{name}")
        }
        manifests = [self.manifest(folder) for folder in folders]
        return sorted(manifests, key=lambda item: item["created_at"], reverse=True)

    def devices(self) -> dict:
        """Device discovery belongs to the evaluator, in its own Python process."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "training.eval.ivritai", "--list-devices"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(503, "Evaluator device discovery timed out") from error
        if result.returncode:
            raise HTTPException(503, f"Evaluator device discovery failed: {result.stderr[-4000:]}")
        return json.loads(result.stdout)

    @staticmethod
    def log_tail(path: Path, limit: int = 32_768) -> str:
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode("utf-8", errors="replace")

    def detail(self, evaluation_id: str) -> dict:
        folder = self.folder(evaluation_id)
        manifest = self.manifest(folder)
        results = []
        results_path = folder / "results.json"
        if results_path.is_file():
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # The evaluator writes this file after each checkpoint; retry on next poll.
        evaluation = None
        evaluation_path = folder / "evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        worker_logs = []
        for job in (evaluation or {}).get("jobs", []):
            if job.get("log") and job["status"] == "running":
                worker_logs.append(
                    {
                        "checkpoint": job["checkpoint"],
                        "device": job["device"],
                        "log": self.log_tail(inside(folder, job["log"]), 8192),
                    }
                )
        return {
            "manifest": manifest,
            "results": results,
            "log": self.log_tail(folder / "evaluation.log"),
            "evaluation": evaluation,
            "worker_logs": worker_logs,
            "artifacts": [
                name
                for name in (
                    "manifest.json",
                    "evaluation.json",
                    "results.json",
                    "evaluation.log",
                    "inputs/sentences.txt",
                    "inputs/args.yaml",
                )
                if (folder / name).is_file()
            ],
            "audio_count": sum(1 for _ in folder.rglob("voice*_sentence*.wav")),
        }

    def start(self, request: EvaluationRequest) -> dict:
        run = self.run_path(request.run_dir)
        available = {item["name"]: Path(item["path"]) for item in self.checkpoints(str(run))}
        if len(set(request.checkpoints)) != len(request.checkpoints):
            raise ValueError("duplicate checkpoints")
        try:
            checkpoints = [available[name] for name in request.checkpoints]
        except KeyError as error:
            raise ValueError(f"checkpoint not found in selected run: {error.args[0]}") from error
        if len(set(request.voices)) != len(request.voices):
            raise ValueError("duplicate voices")
        voices = [inside(self.eval_data, name) for name in request.voices]
        for path in voices:
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError(f"voice file not found or unsupported: {path}")
        sentences = []
        for selection in request.sentences:
            path = inside(self.eval_data, selection.file)
            if path.suffix.lower() != ".txt":
                raise ValueError("sentence files must be UTF-8 .txt files")
            lines = path.read_text(encoding="utf-8").splitlines()
            if selection.line > len(lines) or not lines[selection.line - 1].strip():
                raise ValueError(f"invalid sentence line: {selection.file}:{selection.line}")
            sentences.append({**selection.model_dump(), "text": lines[selection.line - 1].strip()})
        sentences.extend(
            {"file": None, "line": line, "text": text.strip()}
            for line, text in enumerate(request.custom_sentences.splitlines(), 1)
            if text.strip()
        )
        if not sentences:
            raise ValueError("select or enter at least one sentence")

        with self.lock:
            if self.active or self.closing:
                raise HTTPException(
                    409, "An evaluation is already running or the server is closing"
                )
            evaluation_id = "e-" + uuid.uuid4().hex[:8]
            folder = self.output_dir / evaluation_id
            folder.mkdir(parents=True, exist_ok=False)
            inputs = folder / "inputs"
            inputs.mkdir()
            manifest = {
                "schema_version": 1,
                "id": evaluation_id,
                "label": request.label.strip(),
                "status": "running",
                "created_at": now(),
                "started_at": None,
                "finished_at": None,
                "run_dir": str(run),
                "eval_data": str(self.eval_data),
                "output_dir": str(folder),
                "checkpoints": [fingerprint(path) for path in checkpoints],
                "voices": [],
                "sentences": sentences,
                "settings": request.settings.model_dump(),
                "devices": request.devices,
                "expected_audio_count": len(checkpoints) * len(voices) * len(sentences),
                "command": [],
            }
            try:
                shutil.copy2(run / "args.yaml", inputs / "args.yaml")
                voice_snapshots = []
                for index, path in enumerate(voices):
                    snapshot = inputs / f"voice{index + 1:02d}{path.suffix.lower()}"
                    shutil.copy2(path, snapshot)
                    voice_snapshots.append(snapshot)
                    manifest["voices"].append(
                        {
                            **fingerprint(path),
                            "name": request.voices[index],
                            "snapshot": snapshot.relative_to(folder).as_posix(),
                        }
                    )
                sentences_path = inputs / "sentences.txt"
                sentences_path.write_text(
                    "\n".join(item["text"] for item in sentences) + "\n", encoding="utf-8"
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "training.eval.ivritai",
                    str(run),
                    "--sentences",
                    str(sentences_path),
                    "--voices",
                    *map(str, voice_snapshots),
                    "--checkpoints",
                    *map(str, checkpoints),
                    "--output-dir",
                    str(folder),
                ]
                for key, value in request.settings.model_dump().items():
                    flag = "--" + key.replace("_", "-")
                    if isinstance(value, bool):
                        if value:
                            command.append(flag)
                    else:
                        command.extend([flag, str(value)])
                if request.devices:
                    command.extend(["--devices", *request.devices])
                manifest.update(command=command, started_at=now())
                write_json(folder / "manifest.json", manifest)
                with (folder / "evaluation.log").open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
            except Exception as error:
                manifest.update(status="failed", finished_at=now(), error=str(error))
                write_json(folder / "manifest.json", manifest)
                raise
            self.active = evaluation_id
            self.process = process
            self.cancelled = False
            self.worker = threading.Thread(
                target=self._wait, args=(folder, manifest, process), daemon=True
            )
            self.worker.start()
            return manifest

    def export(self, evaluation_id: str) -> Path:
        """Build a standalone snapshot in the OS temporary directory."""
        folder = self.folder(evaluation_id)
        detail = self.detail(evaluation_id)
        temporary = Path(tempfile.mkdtemp(prefix="ivritai-export-"))
        archive = temporary / f"{evaluation_id}.zip"
        try:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                audio_paths = {}

                def add_audio(path: Path) -> str:
                    path = path.resolve()
                    if path not in audio_paths:
                        name = f"audio/clip{len(audio_paths) + 1:05d}{path.suffix.lower()}"
                        output.write(path, name)
                        audio_paths[path] = name
                    return audio_paths[path]

                detail["voice_urls"] = [
                    add_audio(
                        inside(folder, voice["snapshot"])
                        if voice.get("snapshot")
                        else Path(voice["path"])
                    )
                    for voice in detail["manifest"]["voices"]
                ]
                for result in detail["results"]:
                    for record in result.get("records", []):
                        if record.get("audio"):
                            record["audio"] = add_audio(inside(folder, record["audio"]))
                detail["artifacts"] = []
                # JSON lives in an HTML script element: prevent data from closing it.
                data = json.dumps(detail, ensure_ascii=False)
                for char in ("<", ">", "&", "\u2028", "\u2029"):
                    data = data.replace(char, f"\\u{ord(char):04x}")
                html = Path(__file__).with_name("ivritai_server.html").read_text(encoding="utf-8")
                html = html.replace(
                    "<!-- EVALUATION_DATA -->",
                    f'<script id="evaluation-data" type="application/json">{data}</script>',
                )
                output.writestr("index.html", html)
        except Exception:
            shutil.rmtree(temporary)
            raise
        return archive

    def _wait(self, folder: Path, manifest: dict, process: subprocess.Popen) -> None:
        returncode = process.wait()
        if returncode != 0:
            self._kill_process_group(process)
        with self.lock:
            manifest.update(
                status="cancelled"
                if self.cancelled
                else ("completed" if returncode == 0 else "failed"),
                finished_at=now(),
                returncode=returncode,
            )
            write_json(folder / "manifest.json", manifest)
            self.active = None
            self.process = None

    def cancel(self, evaluation_id: str) -> None:
        with self.lock:
            if self.active != evaluation_id or self.process is None:
                raise HTTPException(409, "This evaluation is not running")
            self.cancelled = True
            process = self.process
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                self._kill_process_group(process)
            else:
                process.kill()
        self._kill_process_group(process)

    @staticmethod
    def _kill_process_group(process: subprocess.Popen) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        with self.lock:
            self.closing = True
            active = self.active
        if active:
            try:
                self.cancel(active)
            except HTTPException:
                pass  # The process finished between checking and cancelling.
        if self.worker:
            self.worker.join(timeout=10)


def create_app(
    runs_dir: Path = PROJECT_ROOT / "runs",
    eval_data: Path = PROJECT_ROOT / "eval_data",
    output_dir: Path | None = None,
) -> FastAPI:
    manager = EvaluationManager(runs_dir, eval_data, output_dir or runs_dir / "evals")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        manager.close()

    app = FastAPI(title="ivrit.ai evaluation debugger", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(ValueError)
    async def invalid_input(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("ivritai_server.html"))

    @app.get("/api/catalog")
    def catalog():
        return manager.catalog()

    @app.get("/api/checkpoints")
    def checkpoints(run_dir: str):
        return manager.checkpoints(run_dir)

    @app.get("/api/devices")
    def devices():
        return manager.devices()

    @app.get("/api/voices/{name:path}")
    def voice(name: str):
        path = inside(manager.eval_data, name)
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise FileNotFoundError("voice not found")
        return FileResponse(path)

    @app.get("/api/evaluations")
    def history():
        return manager.history()

    @app.post("/api/evaluations", status_code=201)
    def start(request: EvaluationRequest):
        return manager.start(request)

    @app.get("/api/evaluations/{evaluation_id}")
    def detail(evaluation_id: str):
        return manager.detail(evaluation_id)

    @app.post("/api/evaluations/{evaluation_id}/cancel")
    def cancel(evaluation_id: str):
        manager.cancel(evaluation_id)
        return {"status": "cancelling"}

    @app.post("/api/evaluations/{evaluation_id}/export")
    def export(evaluation_id: str):
        archive = manager.export(evaluation_id)
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=archive.name,
            background=BackgroundTask(shutil.rmtree, archive.parent),
        )

    @app.get("/api/evaluations/{evaluation_id}/files/{name:path}")
    def artifact(evaluation_id: str, name: str):
        path = inside(manager.folder(evaluation_id), name)
        if not path.is_file():
            raise FileNotFoundError("artifact not found")
        return FileResponse(path)

    @app.get("/api/evaluations/{evaluation_id}/voices/{voice_index}")
    def evaluation_voice(evaluation_id: str, voice_index: int):
        folder = manager.folder(evaluation_id)
        voices = manager.manifest(folder)["voices"]
        if not 0 <= voice_index < len(voices):
            raise FileNotFoundError("voice not found")
        voice = voices[voice_index]
        path = inside(folder, voice["snapshot"]) if voice.get("snapshot") else Path(voice["path"])
        if not path.is_file():
            raise FileNotFoundError("voice not found")
        return FileResponse(path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--eval-data", type=Path, default=PROJECT_ROOT / "eval_data")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <runs-dir>/evals")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.runs_dir, args.eval_data, args.output_dir), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
