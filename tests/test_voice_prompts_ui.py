import json
import time

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from hebrew_training.voice_prompts import dedupe
from hebrew_training.voice_prompts.ui import server

RATE = 24_000


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "plenum"
    folder.mkdir()
    words, cursor = [], 0.0
    for letter in "abcdefghijklmn":
        words.append({"word": letter, "start": cursor, "end": cursor + 0.4})
        cursor += 0.4 + (0.4 if letter in "dj" else 0.1)
    rows = []
    for i in range(6):
        sf.write(folder / f"{i}.wav", np.zeros(20 * RATE, dtype="float32"), RATE)
        rows.append(
            {"path": f"{i}.wav", "start": 1.0, "duration": 14.0, "speaker": f"s{i}", "words": words}
        )
    (folder / "train_aligned.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return folder


def wait_for_job(client):
    for _ in range(100):
        state = client.get("/api/job").json()
        if state["state"] != "running":
            return state
        time.sleep(0.1)
    raise AssertionError("extraction did not finish")


def test_extract_twice_then_preview(dataset, tmp_path):
    client = TestClient(server.app)
    out = str(tmp_path / "pool")
    checked = client.post("/api/datasets/check", json={"specs": [str(dataset), "/nope"]}).json()
    assert checked[0]["files"][0]["stage"] == "aligned" and "error" in checked[1]
    assert client.get("/api/target", params={"output": out}).json() == {"exists": False, "count": 0}

    body = {
        "datasets": [str(dataset)],
        "output": out,
        "count": 3,
        "per_speaker": 0,
        "splits": ["train"],
    }
    assert client.post("/api/extract", json=body).status_code == 200
    assert wait_for_job(client)["state"] == "done"
    assert client.get("/api/target", params={"output": out}).json()["count"] == 3

    client.post("/api/extract", json={**body, "count": 2})
    assert wait_for_job(client)["state"] == "done"
    summary = client.get("/api/target", params={"output": out}).json()
    assert summary["count"] == 5 and summary["datasets"] == {"plenum": 5}

    samples = client.get("/api/samples", params={"output": out, "n": 4}).json()
    assert len(samples) == 4
    audio = client.get(samples[0]["audio_url"])
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert (
        client.get("/api/audio", params={"output": out, "file": "../manifest.jsonl"}).status_code
        == 404
    )

    bad = client.post("/api/extract", json={**body, "min_duration": 9})
    assert bad.status_code == 400


def test_similarity_reports_pair_and_percentile(dataset, tmp_path, monkeypatch):
    client = TestClient(server.app)
    out = str(tmp_path / "pool")
    body = {
        "datasets": [str(dataset)],
        "output": out,
        "count": 6,
        "per_speaker": 0,
        "splits": ["train"],
    }
    client.post("/api/extract", json=body)
    wait_for_job(client)

    monkeypatch.setattr(dedupe, "load_prompt_model", lambda settings: object())
    monkeypatch.setattr(
        dedupe,
        "pooled_embedding",
        lambda model, path, space: (
            np.random.default_rng(int(path.stem)).normal(size=8).astype("float32")
        ),
    )
    server._vectors.clear()
    server._references.clear()
    result = client.post("/api/similarity", json={"output": out, "a": 0, "b": 1}).json()
    assert -1 <= result["similarity"] <= 1 and 0 <= result["percentile"] <= 100
    assert result["a"]["id"] == 0 and result["b"]["id"] == 1
    random_pair = client.post("/api/similarity", json={"output": out}).json()
    assert random_pair["a"]["id"] != random_pair["b"]["id"]
    assert client.post("/api/similarity", json={"output": out, "a": 0, "b": 0}).status_code == 400
