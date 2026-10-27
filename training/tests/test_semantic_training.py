"""Semantic conditioning in the training/ trainer: forward, generate, optimizer groups and
checkpoints. CPU only, a fake encoder instead of DictaBERT (tests/test_semantic_conditioning.py).
"""

import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pocket_tts.utils.config import SemanticConditionerConfig  # noqa: E402
from tests.test_semantic_conditioning import (  # noqa: E402
    CORPUS,
    LDIM,
    flow_lm,
    make_sp,
    small_config,
)
from training.args import TrainArgs, load_args  # noqa: E402
from training.checkpointing import EMA, load_checkpoint, save_checkpoint  # noqa: E402
from training.modules.builders import load_model_config  # noqa: E402
from training.modules.model import TrainableTTS  # noqa: E402
from training.modules.samplers import build_flow  # noqa: E402
from training.train_utils import optimizer_param_groups  # noqa: E402

TEXTS = CORPUS[:3]


@pytest.fixture(scope="module")
def sp():
    return make_sp()


def trainable(sp, semantic) -> TrainableTTS:
    args = TrainArgs()
    args.flow_batch_multiplier = 2
    args.text_dropout = args.voice_dropout = 0.0
    return TrainableTTS(flow_lm(sp, semantic), build_flow("lsd"), args)


def batch(sp):
    torch.manual_seed(3)
    tokens = [torch.tensor(sp.encode(t)) for t in TEXTS]
    latents = torch.randn(3, 9, LDIM)
    mask = torch.arange(9)[None, :] < torch.tensor([9, 7, 5])[:, None]
    voice = torch.randn(3, 4, LDIM)
    return latents, mask, tokens, voice


def run_loss(model, sp, **kw):
    torch.manual_seed(11)
    return model(*batch(sp), **kw)


def test_semantic_loss_equals_plain_loss_at_init(sp):
    plain, semantic = trainable(sp, None), trainable(sp, small_config())
    semantic.load_state_dict(plain.state_dict(), strict=False)
    plain.train()
    semantic.train()
    loss_plain, _ = run_loss(plain, sp)
    loss_semantic, metrics = run_loss(semantic, sp, texts=TEXTS)
    torch.testing.assert_close(loss_plain, loss_semantic)
    assert {"semantic_gate", "semantic_cover", "semantic_gate_other"} <= set(metrics)
    assert "semantic_gate_latin" not in metrics  # TEXTS has no Latin letters
    loss_semantic.backward()
    fusion = semantic.flow_lm.semantic_fusion
    assert fusion.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in fusion.parameters())


def test_plain_model_ignores_texts_and_logs_no_semantic_metrics(sp):
    model = trainable(sp, None)
    model.train()
    loss_a, metrics = run_loss(model, sp)
    loss_b, _ = run_loss(model, sp, texts=TEXTS)
    torch.testing.assert_close(loss_a, loss_b)
    assert not any(k.startswith("semantic") for k in metrics)
    assert not any("semantic" in k for k in model.state_dict())


def test_semantic_forward_requires_texts(sp):
    model = trainable(sp, small_config())
    with pytest.raises(ValueError, match="text of every row"):
        run_loss(model, sp)


def test_generate_matches_plain_at_init_and_requires_texts(sp):
    plain, semantic = trainable(sp, None), trainable(sp, small_config())
    semantic.load_state_dict(plain.state_dict(), strict=False)
    tokens = [torch.tensor(sp.encode(t)) for t in TEXTS[:2]]
    voices = [torch.randn(4, LDIM), torch.randn(3, LDIM)]
    kw = dict(max_frames=4, temp=0.0, n_steps=1, cfg_coef=1.0, eos_threshold=1e9)
    torch.manual_seed(5)
    a = plain.generate(tokens, voices, **kw)
    torch.manual_seed(5)
    b = semantic.generate(tokens, voices, texts=TEXTS[:2], **kw)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y)
    with pytest.raises(ValueError, match="texts="):
        semantic.generate(tokens, voices, **kw)


def test_optimizer_groups(sp):
    args = TrainArgs(semantic_lr_scale=4.0)
    plain = trainable(sp, None)
    opt = torch.optim.AdamW(optimizer_param_groups(plain, args), lr=1e-3)
    assert len(opt.param_groups) == 1 and "lr_scale" not in opt.param_groups[0]

    semantic = trainable(sp, small_config())
    opt = torch.optim.AdamW(optimizer_param_groups(semantic, args), lr=1e-3)
    base, fusion = opt.param_groups
    fusion_params = set(map(id, semantic.flow_lm.semantic_fusion.parameters()))
    assert {id(p) for p in fusion["params"]} == fusion_params
    assert not {id(p) for p in base["params"]} & fusion_params
    assert fusion["lr_scale"] == 4.0
    for group in opt.param_groups:  # what the training loop does
        group["lr"] = 2e-4 * group.get("lr_scale", 1.0)
    assert base["lr"] == 2e-4 and fusion["lr"] == pytest.approx(8e-4)


def _save(tmp_path, model, args, sp):
    opt = torch.optim.AdamW(optimizer_param_groups(model, args), lr=1e-3)
    loss, _ = run_loss(model, sp, **_texts(model))
    loss.backward()
    opt.step()
    ema = EMA(model, 0.9)
    save_checkpoint(tmp_path, 7, model, opt, ema, num_keep=2)
    return tmp_path / "checkpoint_00000007.pt"


def _texts(model):
    return {"texts": TEXTS} if model.flow_lm.semantic_fusion is not None else {}


def test_plain_checkpoint_resumes_into_a_semantic_model(sp, tmp_path, caplog):
    args = TrainArgs()
    path = _save(tmp_path, trainable(sp, None), args, sp)
    semantic = trainable(sp, small_config())
    opt = torch.optim.AdamW(optimizer_param_groups(semantic, args), lr=1e-3)
    ema = EMA(semantic, 0.9)
    with caplog.at_level(logging.WARNING):
        step = load_checkpoint(path, semantic, opt, ema)
    assert step == 7
    assert "optimizer starts fresh" in caplog.text
    fusion = semantic.flow_lm.semantic_fusion
    assert torch.equal(fusion.proj.weight, torch.zeros_like(fusion.proj.weight))
    assert "flow_lm.semantic_fusion.proj.weight" in ema.shadow
    # And it trains from there.
    loss, _ = run_loss(semantic, sp, texts=TEXTS)
    loss.backward()
    opt.step()
    ema.update(semantic)


def test_semantic_checkpoint_loads_into_a_plain_model(sp, tmp_path):
    args = TrainArgs()
    semantic = trainable(sp, small_config())
    path = _save(tmp_path, semantic, args, sp)
    plain = trainable(sp, None)
    assert load_checkpoint(path, plain) == 7  # weights only, as training/eval/load_run does
    torch.testing.assert_close(plain.flow_lm.out_eos.weight, semantic.flow_lm.out_eos.weight)


def test_semantic_checkpoint_resumes_into_a_semantic_model(sp, tmp_path):
    args = TrainArgs()
    first = trainable(sp, small_config())
    path = _save(tmp_path, first, args, sp)
    second = trainable(sp, small_config())
    opt = torch.optim.AdamW(optimizer_param_groups(second, args), lr=1e-3)
    load_checkpoint(path, second, opt, EMA(second, 0.9))
    torch.testing.assert_close(
        second.flow_lm.semantic_fusion.proj.weight, first.flow_lm.semantic_fusion.proj.weight
    )
    assert opt.state  # the optimizer state came back


def test_checkpoint_records_its_encoder(sp, tmp_path):
    args = TrainArgs()
    path = _save(tmp_path, trainable(sp, small_config(encoder="renikud_plus")), args, sp)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["semantic_encoder"]["encoder"] == "renikud_plus"
    assert payload["semantic_encoder"]["filename"] == "model.onnx"
    from pocket_tts.utils.weights_loading import read_training_checkpoint

    assert read_training_checkpoint(path)[2] == payload["semantic_encoder"]
    # A plain run adds no key: its checkpoints are unchanged.
    plain_path = _save(tmp_path / "plain", trainable(sp, None), args, sp)
    assert "semantic_encoder" not in torch.load(plain_path, map_location="cpu", weights_only=True)
    # The pocket-tts export carries it as metadata.
    from pocket_tts.conditioners.semantic import read_encoder_record
    from training.checkpointing import export_pocket_safetensors

    model = trainable(sp, small_config(encoder="renikud_plus"))
    export = tmp_path / "model.safetensors"
    export_pocket_safetensors(export, model.flow_lm, torch.nn.Linear(1, 1))
    assert read_encoder_record(export)["encoder"] == "renikud_plus"


def test_checkpoint_from_another_encoder_is_refused(sp, tmp_path):
    args = TrainArgs()
    dictabert_path = _save(tmp_path / "d", trainable(sp, small_config()), args, sp)
    renikud_path = _save(
        tmp_path / "r", trainable(sp, small_config(encoder="renikud_plus")), args, sp
    )
    for path, other in ((dictabert_path, "renikud_plus"), (renikud_path, None)):
        target = trainable(sp, small_config(**({"encoder": other} if other else {})))
        before = target.flow_lm.semantic_fusion.proj.weight.clone()
        opt = torch.optim.AdamW(optimizer_param_groups(target, args), lr=1e-3)
        with pytest.raises(ValueError, match="trained with encoder"):
            load_checkpoint(path, target, opt, EMA(target, 0.9))
        assert torch.equal(target.flow_lm.semantic_fusion.proj.weight, before)
    # Same encoder resumes, with the record in place.
    same = trainable(sp, small_config(encoder="renikud_plus"))
    assert load_checkpoint(renikud_path, same) == 7
    # A checkpoint written before the record existed (no key) is DictaBERT's.
    legacy = torch.load(dictabert_path, map_location="cpu", weights_only=True)
    del legacy["semantic_encoder"]
    torch.save(legacy, tmp_path / "legacy.pt")
    assert load_checkpoint(tmp_path / "legacy.pt", trainable(sp, small_config())) == 7
    with pytest.raises(ValueError, match="trained with encoder dictabert"):
        load_checkpoint(tmp_path / "legacy.pt", trainable(sp, small_config(encoder="renikud_plus")))
    # A plain model still reads any semantic checkpoint (the tensors are dropped).
    assert load_checkpoint(renikud_path, trainable(sp, None)) == 7


def test_model_config_gets_the_semantic_section():
    english = "pocket_tts/config/english.yaml"
    assert load_model_config(english, {}).flow_lm.semantic_conditioner is None
    assert load_model_config(english, {}, None).flow_lm.semantic_conditioner is None
    config = load_model_config(english, {}, {"first_layer": 3, "gate": "vector"})
    section = config.flow_lm.semantic_conditioner
    assert (section.first_layer, section.last_layer, section.gate) == (3, 20, "vector")
    assert load_model_config(english, {}, {}).flow_lm.semantic_conditioner is not None


def test_trainargs_default_is_off():
    args = TrainArgs()
    assert args.semantic_conditioner is None
    with pytest.raises(ValueError, match="semantic_lr_scale"):
        TrainArgs(semantic_lr_scale=0)


def test_multi_dataset_semantic_diagnostic_config():
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_args(configs / "finetune_hebrew_multi_1.yaml")
    args = load_args(configs / "finetune_hebrew_multi_1_semantic_test.yaml")
    assert args.model_config == base.model_config
    assert args.model_overrides == base.model_overrides
    assert args.start_from_pretrained and args.reset_text_embedding
    assert args.data.datasets == base.data.datasets
    assert args.data.curriculum == [{**base.data.curriculum[0], "steps": 1000}]
    for name in ("alt_words_sample_prob", "alt_word_swap_prob", "nikud_word_prob"):
        assert getattr(args.data, name) == getattr(base.data, name)
    assert args.flow == base.flow
    assert args.flow_batch_multiplier == base.flow_batch_multiplier
    assert args.semantic_conditioner is not None
    assert SemanticConditionerConfig(**args.semantic_conditioner) == SemanticConditionerConfig()
    assert args.semantic_lr_scale == 3.0
    assert args.max_steps == args.sample_freq == args.valid_freq == args.ckpt_freq == 1000
    assert 0 < args.optim.warmup_steps < args.max_steps
    assert not args.compile
    assert args.run_dir != base.run_dir
    assert args.batch_size * args.grad_accum_steps * 8 == 64
    assert set(base.sample_sentences) <= set(args.sample_sentences)


def test_multi_dataset_semantic_50k_config():
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_args(configs / "finetune_hebrew_multi_1.yaml")
    diagnostic = load_args(configs / "finetune_hebrew_multi_1_semantic_test.yaml")
    args = load_args(configs / "finetune_hebrew_multi_1_semantic_50k.yaml")
    assert args.max_steps == 50000
    assert args.data.curriculum == base.data.curriculum[:-1] + [
        {**base.data.curriculum[-1], "steps": 5000}
    ]
    assert args.data.datasets == diagnostic.data.datasets
    assert args.model_config == diagnostic.model_config
    assert args.model_overrides == diagnostic.model_overrides
    assert args.start_from_pretrained and args.reset_text_embedding
    assert args.semantic_conditioner == diagnostic.semantic_conditioner
    assert args.semantic_lr_scale == diagnostic.semantic_lr_scale
    for name in ("alt_words_sample_prob", "alt_word_swap_prob", "nikud_word_prob"):
        assert getattr(args.data, name) == getattr(diagnostic.data, name)
    assert args.optim == base.optim
    assert args.flow == diagnostic.flow
    assert args.flow_batch_multiplier == diagnostic.flow_batch_multiplier
    assert not args.compile
    assert args.batch_size * args.grad_accum_steps * 8 == 64
    assert args.sample_freq == args.valid_freq == args.ckpt_freq == 1000
    assert args.num_ckpt_keep >= args.max_steps // args.ckpt_freq
    assert args.sample_sentences == diagnostic.sample_sentences
    assert args.run_dir not in (base.run_dir, diagnostic.run_dir)
