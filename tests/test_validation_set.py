"""The cached validation set (deploy/validation_set.py), including a forced regeneration with a
chosen voice, and the /validation endpoints around it. No model: the generator is a fake."""

import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from deploy import hebrew_tts_server
from deploy.validation_set import ValidationBusy, ValidationSet

SENTENCES = [{"id": f"s{i}", "category": "plain", "text": f"משפט {i}"} for i in range(3)]
VOICES = ["a.wav", "b.wav"]


class Fixture:
    def __init__(self, tmp_path, gate=None):
        path = tmp_path / "sentences.jsonl"
        path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in SENTENCES))
        self.calls: list[tuple[str, str, int]] = []
        self.generation = 0
        self.gate = gate
        self.key = "abcd-cpu"
        self.set = ValidationSet(
            path,
            tmp_path / "cache",
            {"temp": 0.3},
            voices=lambda: list(VOICES),
            generate=self.generate,
            cache_key=lambda: self.key,
        )

    def generate(self, text, voice, seed):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append((text, voice, seed))
        # The bytes say which regeneration made them.
        return f"audio-{voice}-{self.generation}".encode(), 1.0, False

    def wait(self, timeout=5):
        deadline = time.time() + timeout
        while self.set.status()["state"] == "running":
            assert time.time() < deadline, "validation run did not finish"
            time.sleep(0.01)
        return self.set.status()

    def audio(self, item_id="s0"):
        return self.set.audio_path(item_id).read_bytes()


def test_a_plain_start_picks_a_voice_and_a_second_one_reuses_the_cache(tmp_path):
    fx = Fixture(tmp_path)
    fx.set.start()
    status = fx.wait()
    assert status["state"] == "done" and status["voice"] in VOICES
    assert len(fx.calls) == 3
    fx.set.start()
    assert fx.wait()["state"] == "done" and len(fx.calls) == 3  # nothing rendered again


def test_force_replaces_the_cache_with_the_chosen_voice(tmp_path):
    fx = Fixture(tmp_path)
    fx.set.start()
    fx.wait()
    old_voice = fx.set.status()["voice"]
    other = next(v for v in VOICES if v != old_voice)
    fx.generation = 1
    fx.calls.clear()
    fx.set.start(force=True, voice=other)
    status = fx.wait()
    assert status["state"] == "done" and status["voice"] == other
    assert [c[1] for c in fx.calls] == [other] * 3  # every sentence again, with that voice
    assert fx.audio() == f"audio-{other}-1".encode()
    # It is the model's cache now: a later start does not render again, and a restart of the
    # server (a new object over the same directory) shows the chosen voice.
    fx.set.start()
    assert len(fx.calls) == 3
    again = Fixture(tmp_path).set.status()
    assert again["state"] == "done" and again["voice"] == other


def test_force_with_the_same_voice_still_renders_again(tmp_path):
    fx = Fixture(tmp_path)
    fx.set.start()
    fx.wait()
    voice = fx.set.status()["voice"]
    fx.generation = 7
    fx.set.start(force=True, voice=voice)
    fx.wait()
    assert fx.audio() == f"audio-{voice}-7".encode()


def test_force_deletes_stale_audio_of_the_old_set(tmp_path):
    fx = Fixture(tmp_path)
    fx.set.start()
    fx.wait()
    directory = tmp_path / "cache" / fx.key
    stray = directory / "gone.wav"  # e.g. a sentence that was later removed from the file
    stray.write_bytes(b"old")
    fx.set.start(force=True, voice="a.wav")
    fx.wait()
    assert not stray.exists()
    assert sorted(p.name for p in directory.glob("*.wav")) == ["s0.wav", "s1.wav", "s2.wav"]


def test_force_needs_a_known_voice_and_leaves_the_cache_alone(tmp_path):
    fx = Fixture(tmp_path)
    fx.set.start()
    fx.wait()
    before = fx.audio()
    for voice in (None, "", "nope.wav"):
        with pytest.raises(ValueError, match="voice"):
            fx.set.start(force=True, voice=voice)
    assert fx.set.status()["state"] == "done" and fx.audio() == before


def test_force_is_refused_while_a_run_is_in_progress(tmp_path):
    gate = threading.Event()
    fx = Fixture(tmp_path, gate)
    fx.set.start()
    assert fx.set.status()["state"] == "running"
    with pytest.raises(ValidationBusy):
        fx.set.start(force=True, voice="a.wav")
    gate.set()
    assert fx.wait()["state"] == "done"
    assert len(fx.calls) == 3  # the run was not restarted or cut short


def test_force_before_the_model_is_loaded_does_nothing(tmp_path):
    fx = Fixture(tmp_path)
    fx.key = None
    assert fx.set.start(force=True, voice="a.wav")["state"] == "warming"
    assert fx.calls == []


# ------------------------------------------------------------------------------ the endpoint


@pytest.fixture
def client(tmp_path):
    fx = Fixture(tmp_path)
    loader = SimpleNamespace(engine=object(), error=None, stage="ready", model_sha256="ab" * 32)
    app = hebrew_tts_server.create_app(loader, fx.set, SimpleNamespace(), SimpleNamespace())
    return TestClient(app), fx, loader


def test_endpoint_plain_start_still_takes_no_body(client):
    http, fx, _ = client
    response = http.post("/validation/start")
    assert response.status_code == 200
    fx.wait()
    assert http.get("/validation").json()["state"] == "done"


def test_endpoint_force_with_the_selected_voice(client):
    http, fx, _ = client
    http.post("/validation/start")
    fx.wait()
    fx.generation = 3
    response = http.post("/validation/start", data={"force": "true", "voice": "b.wav"})
    assert response.status_code == 200
    fx.wait()
    status = http.get("/validation").json()
    assert status["voice"] == "b.wav" and status["done"] == 3
    assert http.get("/validation/audio/s1.wav").content == b"audio-b.wav-3"


def test_endpoint_force_errors(client):
    http, fx, loader = client
    assert http.post("/validation/start", data={"force": "true"}).status_code == 400
    bad = http.post("/validation/start", data={"force": "true", "voice": "nope.wav"})
    assert bad.status_code == 400 and "nope.wav" in bad.json()["detail"]
    gate = threading.Event()
    fx.gate = gate
    http.post("/validation/start")
    busy = http.post("/validation/start", data={"force": "true", "voice": "a.wav"})
    assert busy.status_code == 409
    gate.set()
    fx.wait()
    loader.engine = None
    assert (
        http.post("/validation/start", data={"force": "true", "voice": "a.wav"}).status_code == 503
    )


def test_page_has_the_regenerate_button(tmp_path):
    fx = Fixture(tmp_path)
    loader = SimpleNamespace(engine=object(), error=None, stage="ready", model_sha256="ab" * 32)
    downloads = SimpleNamespace(html=lambda: "")
    http = TestClient(hebrew_tts_server.create_app(loader, fx.set, downloads, SimpleNamespace()))
    page = http.get("/").text
    assert 'id="valRedo"' in page and "צור מחדש עם הקול הנבחר" in page
    # It sends the selected voice, and a regenerated set must not play from the browser cache.
    assert 'f.append("force", "true"); f.append("voice", voice)' in page
    assert '.wav?t=" + Date.now()' in page
