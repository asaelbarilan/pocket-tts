import hashlib
import json

import pytest

from deploy.model_store import LOCAL_PATHS, MANIFEST, WEIGHTS, ModelStore, copy_verified


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(store, weights=b"weights", config=b"cfg: 1", tokenizer=b"tok"):
    contents = {WEIGHTS: weights, "hebrew_24l.yaml": config, "tokenizer.model": tokenizer}
    model = store / sha(weights)
    model.mkdir(parents=True)
    for name, data in contents.items():
        (model / name).write_bytes(data)
    manifest = {"files": {n: {"sha256": sha(d), "size": len(d)} for n, d in contents.items()}}
    (model / MANIFEST).write_text(json.dumps(manifest))
    return sha(weights)


def test_fetch_copies_to_the_config_paths_and_verifies(tmp_path):
    store, root = tmp_path / "store", tmp_path / "root"
    model_sha = publish(store)
    model = ModelStore(store, model_sha.upper(), root)
    model.fetch_small()
    assert not (root / LOCAL_PATHS[WEIGHTS]).exists()
    model.fetch_weights()
    assert (root / "artifacts/model.safetensors").read_bytes() == b"weights"
    assert (root / "artifacts/hebrew_24l.yaml").read_bytes() == b"cfg: 1"
    assert (root / "tokenizers/tokenizer.model").read_bytes() == b"tok"


def test_corrupt_weights_are_rejected_and_leave_nothing(tmp_path):
    store, root = tmp_path / "store", tmp_path / "root"
    model_sha = publish(store)
    (store / model_sha / WEIGHTS).write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        ModelStore(store, model_sha, root).fetch_weights()
    assert list((root / "artifacts").iterdir()) == []


def test_unpublished_model_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not in the store"):
        ModelStore(tmp_path, sha(b"nope"), tmp_path / "root")


def test_manifest_must_match_its_directory(tmp_path):
    store = tmp_path / "store"
    model_sha = publish(store)
    (store / model_sha).rename(store / sha(b"other"))
    with pytest.raises(ValueError, match="not the directory name"):
        ModelStore(store, sha(b"other"), tmp_path / "root")


def test_pin_must_be_a_sha256(tmp_path):
    with pytest.raises(ValueError, match="not a sha256"):
        ModelStore(tmp_path, "latest", tmp_path)


def test_copy_verified_overwrites_a_stale_copy(tmp_path):
    source, dest = tmp_path / "a", tmp_path / "out" / "b"
    source.write_bytes(b"new")
    dest.parent.mkdir()
    dest.write_bytes(b"old")
    copy_verified(source, dest, sha(b"new"))
    assert dest.read_bytes() == b"new"
