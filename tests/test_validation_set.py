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
        self.settings_calls: list[dict] = []
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

    def generate(self, text, voice, seed, settings=None):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append((text, voice, seed))
        if settings is not None:
            self.settings_calls.append(dict(settings))
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
    assert 'eos_countdown:"eos countdown"' in page
    assert 'id="vwave"' in page and 'id="vplayer"' in page and 'id="vplaySlice"' in page
    assert (
        "How <code>eos threshold</code> and <code>eos countdown</code> control the ending:" in page
    )


def test_initial_and_forced_start_use_configured_voice_and_settings(tmp_path):
    fx = Fixture(tmp_path)
    loader = SimpleNamespace(engine=object(), error=None, stage="ready", model_sha256="ab" * 32)
    http = TestClient(
        hebrew_tts_server.create_app(loader, fx.set, SimpleNamespace(), SimpleNamespace())
    )
    res = http.post(
        "/validation/start",
        data={"voice": "b.wav", "eos_threshold": "4.0", "eos_countdown": "3", "seed": "77"},
    )
    assert res.status_code == 200
    status = fx.wait()
    assert status["voice"] == "b.wav" and status["seed"] == 77
    assert status["settings"]["eos_threshold"] == 4.0 and status["settings"]["eos_countdown"] == 3
    assert all(c[1:] == ("b.wav", 77) for c in fx.calls)
    assert all(s["eos_threshold"] == 4.0 and s["eos_countdown"] == 3 for s in fx.settings_calls)
    # Force regenerate with another voice and updated settings.
    fx.calls.clear()
    fx.settings_calls.clear()
    res2 = http.post(
        "/validation/start",
        data={
            "force": "true",
            "voice": "a.wav",
            "eos_threshold": "2.5",
            "eos_countdown": "2",
            "temp": "0.5",
        },
    )
    assert res2.status_code == 200
    status2 = fx.wait()
    assert status2["voice"] == "a.wav"
    assert status2["settings"]["eos_threshold"] == 2.5
    assert status2["settings"]["eos_countdown"] == 2
    assert status2["settings"]["temp"] == 0.5
    assert all(
        s["eos_threshold"] == 2.5 and s["eos_countdown"] == 2 and s["temp"] == 0.5
        for s in fx.settings_calls
    )


def test_voice_audio_endpoint_serves_known_voices(tmp_path):
    fx = Fixture(tmp_path)
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "a.wav").write_bytes(b"RIFF-fake-wav")
    loader = SimpleNamespace(
        engine=object(), error=None, stage="ready", model_sha256="ab" * 32, voices_dir=voices_dir
    )
    http = TestClient(
        hebrew_tts_server.create_app(loader, fx.set, SimpleNamespace(), SimpleNamespace())
    )
    ok = http.get("/voices/a.wav")
    assert ok.status_code == 200 and ok.content == b"RIFF-fake-wav"
    assert http.get("/voices/missing.wav").status_code == 404


def test_model_metadata_and_preview_banner_show_run_config_and_sha(tmp_path):
    cfg = tmp_path / "hebrew_24l.yaml"
    cfg.write_text(
        "# run: finetune_hebrew_semantic_both_90k\n"
        "# run_config: training/configs/finetune_hebrew_semantic_both_90k.yaml\n"
        "weights_path: artifacts/model.safetensors\n",
        encoding="utf-8",
    )
    meta = hebrew_tts_server.extract_model_metadata(cfg, None, "4b" * 32)
    assert meta["model_run"] == "finetune_hebrew_semantic_both_90k"
    assert meta["model_config"] == "training/configs/finetune_hebrew_semantic_both_90k.yaml"
    assert meta["model_sha256"] == "4b" * 32

    fx = Fixture(tmp_path)
    loader = SimpleNamespace(
        config=cfg,
        store=None,
        engine=SimpleNamespace(device="cuda", n_layers=24, sample_rate=24000, config=cfg),
        error=None,
        stage="ready",
        model_sha256="4b" * 32,
        voices_dir=tmp_path,
        semantic_encoder_info=lambda: {
            "encoders": [{"encoder": "dictabert"}, {"encoder": "renikud_plus"}]
        },
    )
    downloads = SimpleNamespace(html=lambda: "")
    http = TestClient(hebrew_tts_server.create_app(loader, fx.set, downloads, SimpleNamespace()))
    health = http.get("/health").json()
    assert health["model_run"] == "finetune_hebrew_semantic_both_90k"
    assert health["model_config"] == "training/configs/finetune_hebrew_semantic_both_90k.yaml"
    assert health["model_sha256"] == "4b" * 32
    page = http.get("/").text
    assert 'id="modelCard"' in page
    assert "finetune_hebrew_semantic_both_90k" in page
    assert "training/configs/finetune_hebrew_semantic_both_90k.yaml" in page
    assert "4b" * 32 in page


def _old_run(fx, folder, *, sha="1" * 16, sentences=True, wavs=("s0", "s1"), model_run=None):
    """A cache folder an earlier model left behind."""
    directory = fx.set.cache_root / folder
    directory.mkdir(parents=True)
    items = {}
    for i in wavs:
        (directory / f"{i}.wav").write_bytes(f"old-{folder}-{i}".encode())
        items[i] = {"id": i, "status": "ok", "file": f"{i}.wav", "audio_seconds": 1.0}
    manifest = {
        "fingerprint": fx.set.fingerprint,
        "model_sha256": sha,
        "voice": "b.wav",
        "seed": 1234,
        "settings": {"temp": 0.3},
        "items": items,
    }
    if sentences:
        manifest["sentences_sha"] = fx.set._sentences_sha
    else:
        manifest["fingerprint"] = fx.set.fingerprint  # legacy manifests have only this
    if model_run:
        manifest["model_run"] = model_run
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_past_runs_are_discovered_by_set_and_named_by_run_folder_or_date(tmp_path):
    names = {"1" * 16: {"run": "finetune_x", "note": "n"}}
    fx = Fixture(tmp_path)
    fx.set._describe = lambda sha: names.get(sha, {})
    _old_run(fx, "1111111111111111-cuda", sha="1" * 16)
    _old_run(fx, "50k_2222222222222222-cuda", sha="2" * 16, sentences=False)
    _old_run(fx, "3333333333333333-cpu", sha="3" * 16, model_run="stored_run")
    other = _old_run(fx, "4444444444444444-cuda", sha="4" * 16)
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["sentences_sha"] = "different"
    (other / "manifest.json").write_text(json.dumps(manifest))
    (fx.set.cache_root / "5555555555555555-cuda").mkdir()  # no manifest
    _old_run(fx, "abcd-cpu")  # the loaded model's own key is not a past run
    fx.set.start()
    fx.wait()
    _old_run(fx, "abcd-cuda", sha="a" * 16)  # same model, other device kind: a past run

    runs = {r["id"]: r for r in fx.set.past_runs()}
    assert set(runs) == {
        "1111111111111111-cuda",
        "50k_2222222222222222-cuda",
        "3333333333333333-cpu",
        "abcd-cuda",
    }
    assert runs["1111111111111111-cuda"]["label"] == "finetune_x"
    assert runs["1111111111111111-cuda"]["device"] == "cuda"
    assert runs["50k_2222222222222222-cuda"]["label"] == "50k_2222222222222222-cuda"
    assert runs["3333333333333333-cpu"]["label"] == "stored_run"
    assert runs["abcd-cuda"]["state"] == "partial" and runs["abcd-cuda"]["done"] == 2
    assert runs["abcd-cuda"]["date"].endswith("UTC")


def test_run_detail_and_audio_stay_inside_known_runs(tmp_path):
    fx = Fixture(tmp_path)
    _old_run(fx, "1111111111111111-cuda")
    detail = fx.set.run_detail("1111111111111111-cuda")
    assert [i["id"] for i in detail["items"]] == ["s0", "s1"]
    path = fx.set.run_audio_path("1111111111111111-cuda", "s0")
    assert path.read_bytes() == b"old-1111111111111111-cuda-s0"
    assert fx.set.run_audio_path("1111111111111111-cuda", "s2") is None
    assert fx.set.run_audio_path("1111111111111111-cuda", "../x") is None
    for bad in ("..", ".", "", "../cache", "abcd-cpu", "nope"):
        assert fx.set.run_detail(bad) is None
        assert fx.set.run_audio_path(bad, "s0") is None


def test_validation_runs_endpoints(tmp_path):
    fx = Fixture(tmp_path)
    _old_run(fx, "1111111111111111-cuda")
    loader = SimpleNamespace(
        config=None,
        store=None,
        engine=SimpleNamespace(),
        error=None,
        stage="ready",
        model_sha256="a" * 64,
        voices_dir=tmp_path,
    )
    downloads = SimpleNamespace(html=lambda: "")
    http = TestClient(hebrew_tts_server.create_app(loader, fx.set, downloads, SimpleNamespace()))
    listing = http.get("/validation/runs").json()
    assert [s["id"] for s in listing["sentences"]] == ["s0", "s1", "s2"]
    assert [r["id"] for r in listing["runs"]] == ["1111111111111111-cuda"]
    assert len(http.get("/validation/runs/1111111111111111-cuda").json()["items"]) == 2
    audio = http.get("/validation/runs/1111111111111111-cuda/audio/s1.wav")
    assert audio.status_code == 200 and audio.content == b"old-1111111111111111-cuda-s1"
    assert http.get("/validation/runs/nope").status_code == 404
    assert http.get("/validation/runs/1111111111111111-cuda/audio/s2.wav").status_code == 404
    assert 'id="valCmp"' in http.get("/").text


def test_describe_stored_model_reads_run_from_manifest_or_note(tmp_path):
    def publish(sha, manifest):
        (tmp_path / sha).mkdir()
        (tmp_path / sha / "manifest.json").write_text(json.dumps(manifest))

    publish("aa" * 32, {"run": "explicit_run", "note": "x", "created": "2026-01-01T00:00:00Z"})
    publish("bb" * 32, {"note": "finetune_hebrew_foo_90k step 90000"})
    publish("cc" * 32, {"note": "just words"})
    describe = hebrew_tts_server.describe_stored_model
    assert describe(tmp_path, "aa" * 8)["run"] == "explicit_run"
    assert describe(tmp_path, "bb" * 8)["run"] == "finetune_hebrew_foo_90k"
    assert describe(tmp_path, "cc" * 8) == {"run": None, "note": "just words", "created": None}
    assert describe(tmp_path, "dd" * 8, {"run": "fb"}) == {"run": "fb"}
    assert describe(None, "aa" * 8) == {}
