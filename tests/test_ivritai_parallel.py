"""Exercise real spawned workers and result aggregation without loading TTS weights."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from itertools import pairwise
from types import ModuleType, SimpleNamespace

import pytest

from training.eval import ivritai


def fake_generate(checkpoint, sentences, args, asr, progress):
    if "fail" in checkpoint.name:
        raise RuntimeError("intentional inference failure")
    time.sleep(0.5 if checkpoint.stem.endswith("00000100") else 0.04)
    folder = args.output_dir / "same-tag"
    folder.mkdir()
    records = []
    for vi, _voice in enumerate(args.voices):
        for si, text in enumerate(sentences):
            wav = folder / f"voice{vi + 1:02d}_sentence{si + 1:03d}.wav"
            wav.write_bytes(b"fake audio for " + checkpoint.name.encode())
            records.append(
                {
                    "voice_index": vi,
                    "sentence_index": si,
                    "text": text,
                    "audio": wav.relative_to(args.output_dir).as_posix(),
                    "generation_seed": args.seed,
                }
            )
            progress(len(records))
    logging.getLogger("fake_generation").info("generated %s", checkpoint.name)
    return {"checkpoint": str(checkpoint.resolve()), "records": records, "silent": 0, "no_eos": 0}


def fake_asr(args):
    path = args.output_dir.parents[1] / f"asr-{args.device.replace(':', '-')}.txt"
    with path.open("a", encoding="utf-8") as log:
        log.write("loaded\n")
    return object()


def stub_worker(device, args, sentences, tasks, events):
    """Use the production worker loop, substituting only GPU/inference operations."""
    (args.output_dir / f"ready-{device.replace(':', '-')}.txt").touch()
    workers = json.loads((args.output_dir / "evaluation.json").read_text())["worker_count"]
    deadline = time.monotonic() + 15
    while len(list(args.output_dir.glob("ready-*.txt"))) < workers:
        if time.monotonic() > deadline:
            raise RuntimeError("test worker startup timed out")
        time.sleep(0.02)
    ivritai.torch.cuda.set_device = lambda device: None
    ivritai.torch.cuda.empty_cache = lambda: None
    ivritai.evaluate_checkpoint = fake_generate
    ivritai.load_asr = fake_asr
    ivritai._device_worker(device, args, sentences, tasks, events)


def crashed_worker(device, args, sentences, tasks, events):
    os._exit(23)


def blocking_worker(device, args, sentences, tasks, events):
    index = tasks.get()
    events.put({"event": "started", "index": index, "device": device, "at": ivritai.timestamp()})
    (args.output_dir / f"pid-{device.replace(':', '-')}.txt").write_text(str(os.getpid()))
    time.sleep(60)


@pytest.fixture
def eval_args(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "args.yaml").write_text("model_config: test.yaml\n", encoding="utf-8")
    checkpoints = []
    for step in (100, 200, 300, 400):
        checkpoint = run / f"checkpoint_{step:08d}.pt"
        checkpoint.touch()
        checkpoints.append(checkpoint)
    voice = tmp_path / "voice.wav"
    voice.touch()
    sentences = tmp_path / "sentences.txt"
    sentences.write_text("שלום עולם\nמשפט נוסף\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    return ivritai.parse_args(
        [
            str(run),
            "--sentences",
            str(sentences),
            "--voices",
            str(voice),
            "--checkpoints",
            *map(str, checkpoints),
            "--output-dir",
            str(output),
            "--asr",
            "",
            "--seed",
            "17",
        ]
    )


def test_pool_aggregates_ordered_results_and_reuses_asr(eval_args, monkeypatch):
    monkeypatch.setattr(ivritai, "_device_worker", stub_worker)
    sentences = ivritai.read_sentences(eval_args.sentences)
    assert ivritai.run_evaluation(eval_args, sentences, ["cuda:0", "cuda:1"]) == 0
    metadata = json.loads((eval_args.output_dir / "evaluation.json").read_text())
    results = json.loads((eval_args.output_dir / "results.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["worker_count"] == 2
    assert [result["checkpoint"] for result in results] == list(map(str, eval_args.checkpoints))
    assert all(job["generated"] == job["expected"] == 2 for job in metadata["jobs"])
    assert {job["device"] for job in metadata["jobs"]} == {"cuda:0", "cuda:1"}
    # A device never evaluates two checkpoints at once. Across devices there is overlap.
    for device in ("cuda:0", "cuda:1"):
        jobs = sorted(
            (job for job in metadata["jobs"] if job["device"] == device),
            key=lambda job: job["started_at"],
        )
        for first, second in pairwise(jobs):
            assert first["finished_at"] <= second["started_at"]
        assert (
            eval_args.output_dir / f"asr-{device.replace(':', '-')}.txt"
        ).read_text() == "loaded\n"
    a, b = metadata["jobs"][:2]
    assert a["started_at"] < b["finished_at"] and b["started_at"] < a["finished_at"]
    audio_paths = [record["audio"] for result in results for record in result["records"]]
    assert len(set(audio_paths)) == 8
    assert all((eval_args.output_dir / path).is_file() for path in audio_paths)
    assert all(
        record["generation_seed"] == 17 for result in results for record in result["records"]
    )
    assert all(result["execution"]["device"] for result in results)
    assert all(
        "generated" in (eval_args.output_dir / job["log"]).read_text() for job in metadata["jobs"]
    )


def test_failed_checkpoint_does_not_discard_other_results(eval_args, monkeypatch):
    monkeypatch.setattr(ivritai, "_device_worker", stub_worker)
    failed = eval_args.checkpoints[1].with_name("checkpoint_fail.pt")
    failed.touch()
    eval_args.checkpoints[1] = failed
    assert ivritai.run_evaluation(eval_args, ["שלום"], ["cuda:0", "cuda:1"]) == 1
    metadata = json.loads((eval_args.output_dir / "evaluation.json").read_text())
    results = json.loads((eval_args.output_dir / "results.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["jobs"][1]["status"] == "failed"
    assert "intentional inference failure" in metadata["jobs"][1]["error"]
    assert len(results) == 3
    assert [result["checkpoint"] for result in results] == [
        str(path) for path in eval_args.checkpoints if path != failed
    ]
    assert all(job["status"] == "completed" for job in metadata["jobs"] if job["index"] != 1)


def test_worker_crash_is_detected_without_hanging(eval_args, monkeypatch):
    monkeypatch.setattr(ivritai, "_device_worker", crashed_worker)
    assert ivritai.run_evaluation(eval_args, ["שלום"], ["cuda:0", "cuda:1"]) == 1
    metadata = json.loads((eval_args.output_dir / "evaluation.json").read_text())
    assert metadata["status"] == "failed"
    assert "exited unexpectedly" in metadata["error"]
    assert all(job["status"] == "failed" for job in metadata["jobs"])


def test_serial_cancellation_stops_the_queue(eval_args, monkeypatch):
    monkeypatch.setattr(ivritai, "load_asr", lambda args: None)
    started = []

    def cancel(checkpoint, *args, **kwargs):
        started.append(checkpoint)
        raise ivritai.EvaluationCancelled()

    monkeypatch.setattr(ivritai, "evaluate_checkpoint", cancel)
    assert ivritai.run_evaluation(eval_args, ["שלום"], ["cpu"]) == 130
    assert started == eval_args.checkpoints[:1]
    metadata = json.loads((eval_args.output_dir / "evaluation.json").read_text())
    assert metadata["status"] == "cancelled"
    assert all(job["status"] == "cancelled" for job in metadata["jobs"])


def test_device_resolution_and_cpu_fallback(monkeypatch):
    monkeypatch.setattr(
        ivritai,
        "discover_devices",
        lambda: {"devices": [{"id": name} for name in ["cpu", "cuda:0", "cuda:1"]]},
    )
    assert ivritai.resolve_devices("cuda", None) == ["cuda:0"]
    assert ivritai.resolve_devices("cuda", ["auto"]) == ["cuda:0", "cuda:1"]
    assert ivritai.resolve_devices("cpu", ["cuda:1"]) == ["cuda:1"]
    for invalid in (["cuda", "cuda:0"], ["cuda:2"], ["cpu", "cuda:0"]):
        with pytest.raises(ValueError):
            ivritai.resolve_devices("cuda", invalid)
    monkeypatch.setattr(ivritai, "discover_devices", lambda: {"devices": [{"id": "cpu"}]})
    assert ivritai.resolve_devices("cuda", ["auto"]) == ["cpu"]


def test_device_discovery_reports_names_and_memory(monkeypatch):
    monkeypatch.setattr(ivritai.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ivritai.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        ivritai.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name=f"GPU {index}", total_memory=8 * 1024**3),
    )
    inventory = ivritai.discover_devices()
    assert [device["id"] for device in inventory["devices"]] == ["cpu", "cuda:0", "cuda:1"]
    assert inventory["devices"][2]["name"] == "GPU 1"
    assert inventory["devices"][2]["total_memory_bytes"] == 8 * 1024**3
    assert ivritai.parse_args(["--list-devices"]).list_devices
    with pytest.raises(SystemExit):
        ivritai.parse_args([])


def test_asr_uses_selected_gpu_index(monkeypatch):
    import sys

    calls = []
    whisper = ModuleType("faster_whisper")
    whisper.WhisperModel = lambda name, **kwargs: calls.append((name, kwargs))
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper)
    ivritai.load_asr(
        argparse.Namespace(asr="test-model", device="cuda:3", asr_compute_type="int8_float16")
    )
    assert calls == [
        ("test-model", {"device": "cuda", "device_index": 3, "compute_type": "int8_float16"})
    ]
