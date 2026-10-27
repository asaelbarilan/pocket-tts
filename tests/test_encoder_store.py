"""The encoder store (deploy/encoder_store.py) and how the server picks the encoder of a model.

No torch model and no network: the "encoders" are small byte strings.
"""

import hashlib
import json
import logging

import pytest

from deploy import hebrew_tts_server
from deploy.encoder_store import (
    DICTABERT_FILES,
    ENCODERS_DIR,
    LOCAL_DIR,
    MANIFEST,
    EncoderStore,
    describe,
    semantic_config_of,
)
from pocket_tts.utils.config import (
    DICTABERT_LARGE,
    DICTABERT_LARGE_REVISION,
    RENIKUD_PLUS,
    RENIKUD_PLUS_REVISION,
    SemanticConditionerConfig,
)

DICTA = SemanticConditionerConfig()
RENIKUD = SemanticConditionerConfig(encoder="renikud_plus")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(store, config, salt=b"", names=None, main=None):
    """Write one encoder the way deploy/publish_encoder.sh does; returns its sha256."""
    if config.encoder == "renikud_plus":
        main_name, names = config.filename, names or (config.filename,)
    else:
        main_name, names = "model.safetensors", names or DICTABERT_FILES
    contents = {
        n: (main or b"main-" + config.encoder.encode()) + salt
        if n == main_name
        else (n.encode() + salt)
        for n in names
    }
    main_sha = sha(contents[main_name]) if main_name in contents else sha(b"x")
    directory = store / ENCODERS_DIR / main_sha
    directory.mkdir(parents=True)
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    manifest = {
        "format": 1,
        "encoder": config.encoder,
        "model_name": config.model_name,
        "revision": config.revision,
        "main_file": main_name,
        "files": {n: {"sha256": sha(d), "size": len(d)} for n, d in contents.items()},
    }
    (directory / MANIFEST).write_text(json.dumps(manifest))
    return main_sha


def open_store(tmp_path, config, pin=None):
    return EncoderStore(tmp_path / "store" / ENCODERS_DIR, config, tmp_path / "root", pin)


# ------------------------------------------------------------------- the model's config


def write_config(path, section):
    flow_lm = {"semantic_conditioner": section} if section is not None else {}
    path.write_text(json.dumps({"flow_lm": flow_lm}))  # JSON is YAML
    return path


def test_semantic_config_of(tmp_path):
    assert semantic_config_of(write_config(tmp_path / "a.yaml", None)) is None
    # A section written before the encoder was selectable is DictaBERT's.
    old = {"model_name": DICTABERT_LARGE, "revision": DICTABERT_LARGE_REVISION, "gate": "scalar"}
    assert semantic_config_of(write_config(tmp_path / "b.yaml", old)) == DICTA
    assert semantic_config_of(write_config(tmp_path / "c.yaml", {})) == DICTA
    new = semantic_config_of(write_config(tmp_path / "d.yaml", {"encoder": "renikud_plus"}))
    assert new == RENIKUD
    with pytest.raises(ValueError):
        semantic_config_of(write_config(tmp_path / "e.yaml", {"encoder": "bert"}))


def test_describe():
    assert describe(DICTA) == f"dictabert ({DICTABERT_LARGE}@{DICTABERT_LARGE_REVISION})"
    assert describe(RENIKUD).endswith(f"{RENIKUD_PLUS}@{RENIKUD_PLUS_REVISION}/model.onnx)")


# ---------------------------------------------------------------------- finding it


def test_each_model_finds_its_own_encoder_in_a_shared_store(tmp_path):
    store = tmp_path / "store"
    dicta, renikud = publish(store, DICTA), publish(store, RENIKUD)
    assert open_store(tmp_path, DICTA).sha256 == dicta
    assert open_store(tmp_path, RENIKUD).sha256 == renikud
    # Another revision of the same encoder is another encoder.
    with pytest.raises(FileNotFoundError, match="publish_encoder.sh dictabert"):
        open_store(tmp_path, SemanticConditionerConfig(revision="0" * 40))


def test_unpublished_encoder_names_what_to_publish(tmp_path):
    publish(tmp_path / "store", DICTA)
    with pytest.raises(FileNotFoundError, match="renikud_plus.*publish_encoder.sh renikud_plus"):
        open_store(tmp_path, RENIKUD)
    with pytest.raises(FileNotFoundError, match="publish_encoder.sh"):
        EncoderStore(tmp_path / "nowhere", DICTA, tmp_path, None)  # not even a store


def test_the_other_renikud_file_is_not_the_same_encoder(tmp_path):
    publish(tmp_path / "store", RENIKUD)
    int8 = SemanticConditionerConfig(encoder="renikud_plus", filename="model_int8.onnx")
    with pytest.raises(FileNotFoundError):
        open_store(tmp_path, int8)


def test_an_interrupted_upload_is_never_used(tmp_path):
    main_sha = publish(tmp_path / "store", DICTA)
    (tmp_path / "store" / ENCODERS_DIR / main_sha / MANIFEST).unlink()
    with pytest.raises(FileNotFoundError):
        open_store(tmp_path, DICTA)


def test_two_matches_need_a_pin(tmp_path):
    store = tmp_path / "store"
    first, second = publish(store, DICTA), publish(store, DICTA, salt=b"-other")
    with pytest.raises(ValueError, match="SEMANTIC_ENCODER_SHA256"):
        open_store(tmp_path, DICTA)
    assert open_store(tmp_path, DICTA, pin=second.upper()).sha256 == second
    assert open_store(tmp_path, DICTA, pin=first).sha256 == first


def test_a_pin_cannot_select_another_encoder(tmp_path):
    store = tmp_path / "store"
    dicta = publish(store, DICTA)
    with pytest.raises(ValueError, match="needs renikud_plus"):
        open_store(tmp_path, RENIKUD, pin=dicta)
    with pytest.raises(FileNotFoundError, match="not in the store"):
        open_store(tmp_path, DICTA, pin=sha(b"nope"))
    with pytest.raises(ValueError, match="not a sha256"):
        open_store(tmp_path, DICTA, pin="latest")


def test_manifest_must_be_complete_and_match_its_directory(tmp_path):
    store = tmp_path / "store"
    main_sha = publish(store, DICTA)
    directory = store / ENCODERS_DIR / main_sha
    manifest = json.loads((directory / MANIFEST).read_text())
    del manifest["files"]["vocab.txt"]
    (directory / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lists no vocab.txt"):
        open_store(tmp_path, DICTA)
    publish(tmp_path / "store2", DICTA)
    moved = tmp_path / "store2" / ENCODERS_DIR
    (moved / main_sha).rename(moved / sha(b"other"))
    with pytest.raises(ValueError, match="not the directory name"):
        EncoderStore(moved, DICTA, tmp_path / "root", None)


# ---------------------------------------------------------------------- copying it


def test_fetch_copies_every_file_verified(tmp_path):
    store = tmp_path / "store"
    main_sha = publish(store, DICTA)
    encoder = open_store(tmp_path, DICTA)
    directory = encoder.fetch()
    assert directory == tmp_path / "root" / LOCAL_DIR / main_sha
    assert sorted(p.name for p in directory.iterdir()) == sorted(DICTABERT_FILES)
    assert (directory / "model.safetensors").read_bytes() == b"main-dictabert"
    assert encoder.info() == {
        "encoder": "dictabert",
        "model_name": DICTABERT_LARGE,
        "revision": DICTABERT_LARGE_REVISION,
        "sha256": main_sha,
    }
    encoder.discard()
    assert not directory.exists()


def test_renikud_fetch_copies_only_the_onnx_file(tmp_path):
    publish(tmp_path / "store", RENIKUD)
    directory = open_store(tmp_path, RENIKUD).fetch()
    assert [p.name for p in directory.iterdir()] == ["model.onnx"]


def test_a_corrupt_file_is_rejected_and_leaves_nothing_under_its_name(tmp_path):
    store = tmp_path / "store"
    main_sha = publish(store, DICTA)
    (store / ENCODERS_DIR / main_sha / "vocab.txt").write_bytes(b"tampered")
    encoder = open_store(tmp_path, DICTA)
    with pytest.raises(ValueError, match="vocab.txt: sha256"):
        encoder.fetch()
    assert not (encoder.local_dir / "vocab.txt").exists()


# --------------------------------------------------------------- the server's decision


@pytest.fixture
def server_env(monkeypatch):
    for name in ("SEMANTIC_ENCODER_STORE", "SEMANTIC_ENCODER_SHA256"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_a_model_without_semantic_conditioning_needs_no_encoder(tmp_path, server_env):
    config = write_config(tmp_path / "plain.yaml", None)
    assert hebrew_tts_server.open_encoder_store(config, str(tmp_path / "store")) is None


def test_the_encoder_store_defaults_to_the_model_store(tmp_path, server_env):
    store = tmp_path / "store"
    main_sha = publish(store, RENIKUD)
    config = write_config(tmp_path / "m.yaml", {"encoder": "renikud_plus"})
    found = hebrew_tts_server.open_encoder_store(config, str(store))
    assert found.sha256 == main_sha and found.dir == store / ENCODERS_DIR / main_sha
    # Or anywhere the env says.
    elsewhere = tmp_path / "elsewhere"
    server_env.setenv("SEMANTIC_ENCODER_STORE", str(elsewhere))
    with pytest.raises(SystemExit, match="encoder store"):
        hebrew_tts_server.open_encoder_store(config, str(store))
    publish(elsewhere.parent / "x", RENIKUD)  # same layout, other place
    (elsewhere.parent / "x" / ENCODERS_DIR).rename(elsewhere)
    assert hebrew_tts_server.open_encoder_store(config, str(store)).dir.parent == elsewhere


def test_the_pin_is_read_from_the_env(tmp_path, server_env):
    store = tmp_path / "store"
    first, second = publish(store, DICTA), publish(store, DICTA, salt=b"-2")
    config = write_config(tmp_path / "m.yaml", {})
    with pytest.raises(SystemExit, match="SEMANTIC_ENCODER_SHA256"):
        hebrew_tts_server.open_encoder_store(config, str(store))
    server_env.setenv("SEMANTIC_ENCODER_SHA256", second)
    assert hebrew_tts_server.open_encoder_store(config, str(store)).sha256 == second
    server_env.setenv("SEMANTIC_ENCODER_SHA256", sha(b"nope"))
    with pytest.raises(SystemExit, match="not in the store"):
        hebrew_tts_server.open_encoder_store(config, str(store))
    assert first != second


def test_without_any_store_the_encoder_comes_from_the_hub_cache(tmp_path, server_env, caplog):
    config = write_config(tmp_path / "m.yaml", {})
    with caplog.at_level(logging.INFO, logger="hebrew_tts_server"):
        assert hebrew_tts_server.open_encoder_store(config, None) is None
    assert "Hugging Face cache" in caplog.text


def test_a_pin_without_a_semantic_model_only_warns(tmp_path, server_env, caplog):
    server_env.setenv("SEMANTIC_ENCODER_SHA256", sha(b"x"))
    config = write_config(tmp_path / "plain.yaml", None)
    with caplog.at_level(logging.WARNING, logger="hebrew_tts_server"):
        assert hebrew_tts_server.open_encoder_store(config, str(tmp_path)) is None
    assert "no semantic encoder" in caplog.text


def test_loader_copies_the_encoder_and_passes_its_directory(tmp_path, monkeypatch):
    store = tmp_path / "store"
    publish(store, DICTA)
    encoder = EncoderStore(store / ENCODERS_DIR, DICTA, tmp_path / "root")
    seen = {}

    class FakeEngine:
        def __init__(self, config, voices_dir, device, semantic_encoder_dir=None):
            seen["dir"] = semantic_encoder_dir
            seen["files"] = sorted(p.name for p in semantic_encoder_dir.iterdir())
            self.model = type(
                "M",
                (),
                {
                    "flow_lm": type(
                        "F", (), {"semantic_fusion": type("S", (), {"config": DICTA})()}
                    )()
                },
            )()

    monkeypatch.setattr(hebrew_tts_server, "Engine", FakeEngine)
    config = write_config(tmp_path / "m.yaml", {})
    (tmp_path / "w.safetensors").write_bytes(b"w")
    config.write_text(json.dumps({"weights_path": str(tmp_path / "w.safetensors")}))
    loader = hebrew_tts_server.EngineLoader(
        config, tmp_path, hebrew_tts_server.torch.device("cpu"), None, encoder
    )
    loader._load()
    assert loader.stage == "ready", loader.error
    assert seen["dir"] == encoder.local_dir and seen["files"] == sorted(DICTABERT_FILES)
    assert not encoder.local_dir.exists()  # freed once loaded
    info = loader.semantic_encoder_info()
    assert info["encoder"] == "dictabert" and info["sha256"] == encoder.sha256
