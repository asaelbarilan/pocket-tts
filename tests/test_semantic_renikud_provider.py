"""CUDA is the default; explicit CPU placement preserves frozen encoder identity."""

from types import SimpleNamespace

import pytest
import torch

from pocket_tts.conditioners import semantic_renikud as renikud
from pocket_tts.utils.config import SemanticConditionerConfig


def test_provider_defaults_and_identity():
    cpu = SemanticConditionerConfig(encoder="renikud_plus", onnx_provider="cpu")
    cuda = SemanticConditionerConfig(encoder="renikud_plus")
    assert cpu.onnx_provider == "cpu"
    assert cuda.onnx_provider == "cuda"
    assert SemanticConditionerConfig().onnx_provider is None
    assert cpu.identity() == cuda.identity()
    assert cpu.filename == cuda.filename == "model.onnx"
    with pytest.raises(ValueError, match="renikud_plus"):
        SemanticConditionerConfig(onnx_provider="cuda")
    with pytest.raises(ValueError, match="onnx_provider"):
        SemanticConditionerConfig(encoder="renikud_plus", onnx_provider="auto")


def test_cpu_provider_never_touches_cuda_or_loads_runtime(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("explicit CPU provider must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(renikud, "_need_onnxruntime", unexpected)
    assert renikud._onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_cuda_provider_requires_gpu_runtime(monkeypatch):
    monkeypatch.setattr(
        renikud,
        "_need_onnxruntime",
        lambda: SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )
    with pytest.raises(RuntimeError, match="CPU-only"):
        renikud._onnx_providers("cuda")


def test_cuda_provider_requires_device(monkeypatch):
    monkeypatch.setattr(
        renikud,
        "_need_onnxruntime",
        lambda: SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"]),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA device"):
        renikud._onnx_providers("cuda")


def test_cuda_provider_uses_rank_device_and_full_fp32(monkeypatch):
    preloaded = []
    monkeypatch.setattr(
        renikud,
        "_need_onnxruntime",
        lambda: SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            preload_dlls=lambda: preloaded.append(True),
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert renikud._onnx_providers("cuda") == [
        ("CUDAExecutionProvider", {"device_id": 3, "use_tf32": 0})
    ]
    assert preloaded == [True]


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        renikud._onnx_providers("auto")


@pytest.mark.parametrize(
    "provider,actual",
    [
        ("cpu", "CPUExecutionProvider"),
        ("cuda", "CUDAExecutionProvider"),
        ("cuda", "CPUExecutionProvider"),
    ],
)
def test_session_cannot_silently_fall_back_to_cpu(monkeypatch, provider, actual):
    disabled = []
    session = SimpleNamespace(
        get_providers=lambda: [actual], disable_fallback=lambda: disabled.append(True)
    )
    model = SimpleNamespace(metadata_props=[])
    monkeypatch.setattr(
        renikud,
        "_need_onnx",
        lambda: SimpleNamespace(load=lambda path: model, save=lambda model, path: None),
    )
    monkeypatch.setattr(
        renikud,
        "_need_onnxruntime",
        lambda: SimpleNamespace(
            SessionOptions=SimpleNamespace,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
            InferenceSession=lambda *args, **kwargs: session,
        ),
    )
    monkeypatch.setattr(renikud, "_onnx_providers", lambda provider: [])
    monkeypatch.setattr(renikud, "tap_layer_outputs", lambda model, layers: (model, ["hidden"]))
    if provider == "cuda" and actual == "CPUExecutionProvider":
        with pytest.raises(RuntimeError, match="Refusing to silently"):
            renikud._tapped_session("unused.onnx", [8], 1, provider=provider)
    else:
        result, names, meta = renikud._tapped_session("unused.onnx", [8], 1, provider=provider)
        assert result is session and names == ["hidden"] and meta == {}
        assert disabled == ([True] if provider == "cuda" else [])
