"""Dashboard integration tests with a tiny subprocess in place of model inference."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import wave
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from training.eval import ivritai_server as server


@pytest.fixture
def dashboard(tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "example"
    run.mkdir(parents=True)
    (run / "args.yaml").write_text("model_config: test.yaml\n", encoding="utf-8")
    for step in (100, 200):
        (run / f"checkpoint_{step:08d}.pt").write_bytes(b"checkpoint placeholder")
    data = tmp_path / "eval_data"
    (data / "nested").mkdir(parents=True)
    (data / "sentences.txt").write_text("שלום עולם\n\nמשפט שני\n", encoding="utf-8")
    voice = data / "nested" / "voice.wav"
    with wave.open(str(voice), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\0\0" * 240)
    app = server.create_app(runs, data)
    with TestClient(app) as client:
        yield client, app.state.manager, run, data


def request_payload(run: Path) -> dict:
    return {
        "run_dir": str(run),
        "checkpoints": ["checkpoint_00000100.pt", "checkpoint_00000200.pt"],
        "voices": ["nested/voice.wav"],
        "sentences": [{"file": "sentences.txt", "line": 3}],
        "custom_sentences": "  משפט מותאם  \n\n",
        "label": "test comparison",
        "devices": ["cpu"],
        "settings": {"device": "cpu", "asr_compute_type": "int8", "use_ema": True},
    }


def wait_for_job(client: TestClient, evaluation_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        detail = client.get(f"/api/evaluations/{evaluation_id}").json()
        if detail["manifest"]["status"] != "running":
            return detail
        time.sleep(0.02)
    pytest.fail("evaluation subprocess did not finish")


FAKE_EVALUATOR = """
import argparse, json, shutil
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('run_dir')
p.add_argument('--sentences')
p.add_argument('--voices', nargs='+')
p.add_argument('--checkpoints', nargs='+')
p.add_argument('--output-dir')
a, _ = p.parse_known_args()
root = Path(a.output_dir)
sentences = Path(a.sentences).read_text(encoding='utf-8').splitlines()
results = []
for checkpoint in a.checkpoints:
    folder = root / Path(checkpoint).stem
    folder.mkdir()
    records = []
    for vi, voice in enumerate(a.voices):
        for si, text in enumerate(sentences):
            wav = folder / f'voice{vi+1:02d}_sentence{si+1:03d}.wav'
            shutil.copyfile(voice, wav)
            records.append(dict(voice_index=vi, sentence_index=si, text=text,
                audio=wav.relative_to(root).as_posix(), seconds=0.01,
                silent=False, no_eos=False, generation_seed=0))
    results.append(dict(checkpoint=checkpoint, step=100, weights='ema', records=records,
        silent=0, no_eos=0))
    (root / 'results.json').write_text(json.dumps(results), encoding='utf-8')
print('evaluation completed', flush=True)
"""


def test_discovery_and_voice_playback(dashboard):
    client, manager, run, _ = dashboard
    catalog = client.get("/api/catalog").json()
    assert catalog["runs"] == [{"name": "example", "path": str(run)}]
    assert [item["line"] for item in catalog["sentences"]] == [1, 3]
    assert catalog["voices"][0]["name"] == "nested/voice.wav"
    assert catalog["defaults"]["batch_size"] == 6
    assert catalog["defaults"]["cfg"] == 2.0
    assert catalog["defaults"]["use_ema"] is True
    assert catalog["defaults"]["asr"] == ""
    checkpoints = client.get("/api/checkpoints", params={"run_dir": "example"}).json()
    assert [item["name"] for item in checkpoints] == [
        "checkpoint_00000100.pt",
        "checkpoint_00000200.pt",
    ]
    audio = client.get("/api/voices/nested/voice.wav", headers={"Range": "bytes=0-43"})
    assert audio.status_code == 206
    assert audio.content.startswith(b"RIFF")
    assert client.get("/").status_code == 200
    assert manager.output_dir.name == "evals"


def test_evaluation_snapshots_results_and_restart(dashboard, monkeypatch):
    client, manager, run, data = dashboard
    real_popen = subprocess.Popen
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return real_popen([sys.executable, "-u", "-c", FAKE_EVALUATOR, *command[4:]], **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", launch)
    response = client.post("/api/evaluations", json=request_payload(run))
    assert response.status_code == 201, response.text
    evaluation_id = response.json()["id"]
    detail = wait_for_job(client, evaluation_id)
    manifest = detail["manifest"]
    assert manifest["status"] == "completed"
    assert manifest["returncode"] == 0
    assert manifest["created_at"] <= manifest["started_at"] <= manifest["finished_at"]
    assert detail["audio_count"] == manifest["expected_audio_count"] == 4
    assert len(detail["results"]) == 2
    assert "evaluation completed" in detail["log"]
    assert commands[0][:5] == [sys.executable, "-u", "-m", "training.eval.ivritai", str(run)]
    assert "--use-ema" in commands[0]
    # Check the actual evaluator accepts the dashboard's command, without inference.
    from training.eval.ivritai import parse_args

    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", commands[0][3:])
        parsed = parse_args()
    assert parsed.run_dir == run
    assert parsed.device == "cpu"
    assert parsed.devices == ["cpu"]
    assert parsed.use_ema is True
    assert parsed.batch_size == 6
    assert parsed.cfg == 2.0
    assert parsed.asr == ""
    assert parsed.output_dir == manager.folder(evaluation_id)
    assert manifest["sentences"] == [
        {"file": "sentences.txt", "line": 3, "text": "משפט שני"},
        {"file": None, "line": 1, "text": "משפט מותאם"},
    ]
    assert Path(manifest["checkpoints"][0]["path"]).parent == run
    base = f"/api/evaluations/{evaluation_id}/files/"
    assert client.get(base + "inputs/sentences.txt").text == "משפט שני\nמשפט מותאם\n"
    assert client.get(base + "inputs/args.yaml").text == "model_config: test.yaml\n"
    # Playback and reproducibility use snapshots, even after source assets change.
    (data / "nested/voice.wav").write_bytes(b"changed")
    assert client.get(base + manifest["voices"][0]["snapshot"]).content.startswith(b"RIFF")
    assert client.get(f"/api/evaluations/{evaluation_id}/voices/0").content.startswith(b"RIFF")
    record = detail["results"][0]["records"][0]
    assert client.get(base + record["audio"]).content.startswith(b"RIFF")
    restarted = server.create_app(manager.runs_dir, manager.eval_data)
    with TestClient(restarted) as reopened:
        assert reopened.get("/api/evaluations").json()[0]["id"] == evaluation_id
        assert (
            reopened.get(f"/api/evaluations/{evaluation_id}").json()["results"] == detail["results"]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoints": ["checkpoint_00000999.pt"]},
        {"checkpoints": ["checkpoint_00000100.pt", "checkpoint_00000100.pt"]},
        {"voices": ["../outside.wav"]},
        {"sentences": [{"file": "sentences.txt", "line": 2}]},
        {"sentences": [], "custom_sentences": "\n "},
        {"settings": {"batch_size": 0}},
    ],
)
def test_invalid_selections_do_not_launch(dashboard, changes):
    client, manager, run, _ = dashboard
    response = client.post("/api/evaluations", json={**request_payload(run), **changes})
    assert response.status_code in (400, 422), response.text
    assert manager.active is None
    assert not manager.output_dir.exists()


def test_single_job_cancellation_and_partial_json(dashboard, monkeypatch):
    client, manager, run, _ = dashboard
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda command, **kwargs: real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        ),
    )
    evaluation_id = client.post("/api/evaluations", json=request_payload(run)).json()["id"]
    assert client.post("/api/evaluations", json=request_payload(run)).status_code == 409
    (manager.folder(evaluation_id) / "results.json").write_text("[{", encoding="utf-8")
    assert client.get(f"/api/evaluations/{evaluation_id}").json()["results"] == []
    assert client.post(f"/api/evaluations/{evaluation_id}/cancel").status_code == 200
    assert wait_for_job(client, evaluation_id)["manifest"]["status"] == "cancelled"
    assert client.post(f"/api/evaluations/{evaluation_id}/cancel").status_code == 409
    assert (
        client.get(
            f"/api/evaluations/{evaluation_id}/files/%2E%2E/%2E%2E/example/args.yaml"
        ).status_code
        == 400
    )


def test_subprocess_failure_is_recorded(dashboard, monkeypatch):
    client, _, run, _ = dashboard
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda command, **kwargs: real_popen(
            [sys.executable, "-c", "print('model load failed'); raise SystemExit(2)"], **kwargs
        ),
    )
    evaluation_id = client.post("/api/evaluations", json=request_payload(run)).json()["id"]
    detail = wait_for_job(client, evaluation_id)
    assert detail["manifest"]["status"] == "failed"
    assert detail["manifest"]["returncode"] == 2
    assert "model load failed" in detail["log"]


def test_server_shutdown_stops_active_process(dashboard, monkeypatch):
    client, manager, run, _ = dashboard
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda command, **kwargs: real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        ),
    )
    evaluation_id = client.post("/api/evaluations", json=request_payload(run)).json()["id"]
    process = manager.process
    manager.close()
    assert process.poll() is not None
    assert manager.active is None
    assert (
        client.get(f"/api/evaluations/{evaluation_id}").json()["manifest"]["status"] == "cancelled"
    )


def test_interrupted_job_recovery(tmp_path):
    folder = tmp_path / "runs/evals/e-old"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    manager = server.EvaluationManager(tmp_path / "runs", tmp_path / "eval_data", folder.parent)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["finished_at"]
    assert manager.active is None


def test_device_discovery_delegates_to_evaluator(dashboard, monkeypatch):
    client, _, _, _ = dashboard
    commands = []

    def detect(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"devices": [{"id": "cuda:1", "name": "Test GPU"}]}),
            stderr="",
        )

    monkeypatch.setattr(server.subprocess, "run", detect)
    response = client.get("/api/devices")
    assert response.status_code == 200
    assert response.json()["devices"][0]["name"] == "Test GPU"
    assert commands == [[sys.executable, "-m", "training.eval.ivritai", "--list-devices"]]


def test_dashboard_consumes_standalone_cli_results(dashboard):
    client, manager, run, data = dashboard
    folder = manager.output_dir / "cli-comparison"
    audio_dir = folder / "checkpoints/cp001/tag"
    audio_dir.mkdir(parents=True)
    voice = data / "nested/voice.wav"
    audio = "checkpoints/cp001/tag/voice01_sentence001.wav"
    (folder / audio).write_bytes(voice.read_bytes())
    checkpoint = str(run / "checkpoint_00000100.pt")
    server.write_json(
        folder / "evaluation.json",
        {
            "status": "completed",
            "started_at": server.now(),
            "finished_at": server.now(),
            "run_dir": str(run),
            "settings": {"asr": ""},
            "devices": ["cuda:0"],
            "voices": [str(voice)],
            "sentences": ["שלום"],
            "jobs": [
                {
                    "checkpoint": checkpoint,
                    "status": "completed",
                    "device": "cuda:0",
                    "generated": 1,
                    "expected": 1,
                }
            ],
        },
    )
    (folder / "results.json").write_text(
        json.dumps([{"checkpoint": checkpoint, "records": [{"audio": audio}]}])
    )
    manifest = client.get("/api/evaluations").json()[0]
    assert manifest["id"] == "cli-comparison"
    assert manifest["managed"] is False
    assert manifest["expected_audio_count"] == 1
    detail = client.get("/api/evaluations/cli-comparison").json()
    assert detail["audio_count"] == 1
    assert "evaluation.json" in detail["artifacts"]
    assert "manifest.json" not in detail["artifacts"]
    assert detail["manifest"]["sentences"] == [{"text": "שלום"}]
    assert client.get("/api/evaluations/cli-comparison/voices/0").content.startswith(b"RIFF")
    assert client.get(f"/api/evaluations/cli-comparison/files/{audio}").content.startswith(b"RIFF")
    response = client.post("/api/evaluations/cli-comparison/export")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("audio/clip00001.wav") == voice.read_bytes()
        assert archive.read("audio/clip00002.wav") == (folder / audio).read_bytes()


def saved_export_evaluation(manager, run, data):
    folder = manager.output_dir / "export-test"
    (folder / "inputs").mkdir(parents=True)
    voice = (data / "nested/voice.wav").read_bytes()
    (folder / "inputs/voice01.wav").write_bytes(voice)
    (folder / "generated.wav").write_bytes(voice + b"generated")
    manifest = {
        "id": folder.name,
        "status": "completed",
        "label": "</script><script>alert('label')</script>",
        "started_at": server.now(),
        "finished_at": server.now(),
        "checkpoints": [{"path": str(run / "checkpoint_00000100.pt")}],
        "voices": [{"name": "voice.wav", "snapshot": "inputs/voice01.wav"}],
        "sentences": [{"text": "שלום </script><script>alert('text')</script>"}],
        "expected_audio_count": 1,
    }
    server.write_json(folder / "manifest.json", manifest)
    results = [
        {
            "checkpoint": manifest["checkpoints"][0]["path"],
            "step": 100,
            "weights": "ema",
            "wer": 0.25,
            "cer": 0.1,
            "silent": 0,
            "no_eos": 0,
            "records": [
                {
                    "audio": "generated.wav",
                    "voice_index": 0,
                    "sentence_index": 0,
                    "seconds": 0.01,
                    "hypothesis": "שלום",
                    "generation_seed": 42,
                }
            ],
        }
    ]
    (folder / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return folder


def test_static_export_audio_data_and_temporary_cleanup(dashboard, monkeypatch, tmp_path):
    client, manager, run, data = dashboard
    folder = saved_export_evaluation(manager, run, data)
    temporary = tmp_path / "temporary-export"
    temporary.mkdir()
    monkeypatch.setattr(server.tempfile, "mkdtemp", lambda **kwargs: str(temporary))
    # Managed exports use the saved prompt even if the original changes.
    (data / "nested/voice.wav").write_bytes(b"changed")
    response = client.post("/api/evaluations/export-test/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="export-test.zip"' in response.headers["content-disposition"]
    assert not temporary.exists()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "index.html",
            "audio/clip00001.wav",
            "audio/clip00002.wav",
        }
        html = archive.read("index.html").decode("utf-8")
        start = '<script id="evaluation-data" type="application/json">'
        payload = html.split(start, 1)[1].split("</script>", 1)[0]
        assert "<" not in payload
        exported = json.loads(payload)
        original = manager.detail(folder.name)
        assert exported["manifest"] == original["manifest"]
        record = exported["results"][0]["records"][0]
        assert record["hypothesis"] == "שלום"
        assert exported["results"][0]["wer"] == 0.25
        assert archive.read(record["audio"]) == (folder / "generated.wav").read_bytes()
        assert (
            archive.read(exported["voice_urls"][0]) == (folder / "inputs/voice01.wav").read_bytes()
        )
        assert exported["artifacts"] == []
    assert original["results"][0]["records"][0]["audio"] == "generated.wav"


@pytest.mark.parametrize("audio, status", [("missing.wav", 404), ("../outside.wav", 400)])
def test_static_export_failure_cleans_temporary_files(
    dashboard, monkeypatch, tmp_path, audio, status
):
    client, manager, run, data = dashboard
    folder = saved_export_evaluation(manager, run, data)
    results = json.loads((folder / "results.json").read_text())
    results[0]["records"][0]["audio"] = audio
    (folder / "results.json").write_text(json.dumps(results))
    temporary = tmp_path / "failed-export"
    temporary.mkdir()
    monkeypatch.setattr(server.tempfile, "mkdtemp", lambda **kwargs: str(temporary))
    assert client.post("/api/evaluations/export-test/export").status_code == status
    assert not temporary.exists()
    assert client.post("/api/evaluations/unknown/export").status_code == 404


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cancellation")
def test_dashboard_cancels_orchestrator_and_all_workers(dashboard, monkeypatch):
    client, manager, run, _ = dashboard
    real_popen = subprocess.Popen
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "from training.eval import ivritai; "
        "from test_ivritai_parallel import blocking_worker; "
        "ivritai._device_worker = blocking_worker; "
        "ivritai.resolve_devices = lambda device, devices: ['cuda:0', 'cuda:1']; "
        "raise SystemExit(ivritai.main())"
    )
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda command, **kwargs: real_popen(
            [sys.executable, "-u", "-c", code, *command[4:]], **kwargs
        ),
    )
    payload = {**request_payload(run), "devices": ["cuda:0", "cuda:1"]}
    response = client.post("/api/evaluations", json=payload)
    assert response.status_code == 201, response.text
    evaluation_id = response.json()["id"]
    folder = manager.folder(evaluation_id)
    deadline = time.monotonic() + 20
    while len(list(folder.glob("pid-*.txt"))) != 2:
        if time.monotonic() > deadline:
            pytest.fail(client.get(f"/api/evaluations/{evaluation_id}").json()["log"])
        time.sleep(0.05)
    pids = [int(path.read_text()) for path in folder.glob("pid-*.txt")]
    assert client.post(f"/api/evaluations/{evaluation_id}/cancel").status_code == 200
    detail = wait_for_job(client, evaluation_id)
    assert detail["manifest"]["status"] == "cancelled"
    assert detail["evaluation"]["status"] == "cancelled"
    assert all(job["status"] == "cancelled" for job in detail["evaluation"]["jobs"])
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
