"""Builders: assemble the trainable model, the frozen Mimi codec, and (for
distillation runs) the frozen teacher from a pocket-tts config + TrainArgs."""

import copy
import logging
import math
import typing as tp
from pathlib import Path

import safetensors.torch
import torch
import yaml
from torch import nn

from pocket_tts.conditioners.semantic import (
    ENCODER_RECORD_KEY,
    load_state_dict_allowing_semantic,
    read_encoder_record,
)
from pocket_tts.models.mimi import MimiModel, build_mimi
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.utils.config import Config
from pocket_tts.utils.utils import download_if_necessary

from ..args import TrainArgs
from ..scripts.shrink_checkpoint import shrink
from .model import TrainableTTS
from .samplers import build_flow
from .utils import disable_grad, dit_init, gaussian_init, stamp_state_names

logger = logging.getLogger(__name__)


def _local_tokenizer_vocab_size(tokenizer_path: str) -> int | None:
    """The vocabulary size of a tokenizer on disk, or None for a remote (released) one."""
    if tokenizer_path.startswith(("hf://", "http://", "https://")):
        return None
    tokenizer_file = Path(tokenizer_path)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(
            f"tokenizer {tokenizer_path} does not exist; build it with "
            "`python -m hebrew_training.train_tokenizer` or fix "
            "flow_lm.lookup_table.tokenizer_path"
        )
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_file)).vocab_size()


def load_model_config(
    path: str, overrides: dict[str, tp.Any], semantic: dict[str, tp.Any] | None = None
):
    """A pocket-tts model config with dotted-path fields replaced.

    Lets one released config serve as the architecture for any variant of it
    (the 24-layer teacher is the 6-layer config with num_layers changed), so a
    language switch is one path rather than a forked copy per depth.

    `flow_lm.lookup_table.n_bins` is taken from a local tokenizer file rather
    than from the config: pocket-tts asserts the two are equal, so the
    tokenizer is the source of truth and a vocabulary size is chosen once, when
    the tokenizer is trained. Released tokenizers (hf:// paths) keep the
    config's value.

    `semantic` (TrainArgs.semantic_conditioner) is merged into
    flow_lm.semantic_conditioner, creating the section when the config has none. If it
    selects a different `encoder` than the config's section, it replaces that section.
    """
    config_file = download_if_necessary(str(path))
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    raw = yaml.safe_load(config_file.read_text())
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"{path} has no section {key!r} (from override {dotted!r})")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"{path} has no field {dotted!r} to override")
        node[leaf] = value
    if semantic is not None:
        base = raw["flow_lm"].get("semantic_conditioner") or {}
        if semantic.get("encoder", "dictabert") != base.get("encoder", "dictabert"):
            # The base section's model, revision, file and limits belong to its own encoder,
            # so naming another one replaces the section instead of merging into it.
            logger.info(
                "semantic_conditioner.encoder %s replaces the model config's %s section",
                semantic.get("encoder", "dictabert"),
                base.get("encoder", "dictabert"),
            )
            base = {}
        raw["flow_lm"]["semantic_conditioner"] = {**base, **semantic}
    lookup_table = raw["flow_lm"]["lookup_table"]
    vocab_size = _local_tokenizer_vocab_size(str(lookup_table["tokenizer_path"]))
    if vocab_size is not None:
        if lookup_table.get("n_bins") not in (None, vocab_size):
            logger.info(
                "n_bins %s -> %s, the size of %s",
                lookup_table["n_bins"],
                vocab_size,
                lookup_table["tokenizer_path"],
            )
        lookup_table["n_bins"] = vocab_size
    return Config(**raw)


def attach_distillation(model: TrainableTTS, flow_lm: nn.Module, args: TrainArgs) -> None:
    """Give `model` a frozen teacher and freeze what distillation must not move.

    Two shapes share this path: a separately trained (usually deeper) teacher
    named by distill_teacher_config, or a frozen copy of the model itself when
    only guidance is being baked in.
    """
    if args.text_dropout or args.voice_dropout:
        # The teacher's targets are always fully conditioned, and a distilled
        # student is sampled at cfg 1 with no null branch. Dropping the
        # student's conditioning asks it to predict the conditioned target
        # from a null input.
        logger.warning(
            "distillation with text_dropout=%s voice_dropout=%s: set both to 0",
            args.text_dropout,
            args.voice_dropout,
        )

    if args.distill_teacher_config:
        # Depth distillation: the teacher is a separately trained (usually
        # deeper) model. Same d_model, so the per-frame head and every
        # non-backbone tensor transfers to the student unchanged.
        t_args = copy.deepcopy(args)
        t_args.model_config = args.distill_teacher_config
        t_args.model_overrides = args.distill_teacher_overrides
        t_args.distill_cfg_coef = 0.0
        t_args.start_from_pretrained = False
        teacher = build_models(t_args)[0].flow_lm
        payload = torch.load(args.distill_teacher_weights, map_location="cpu", weights_only=True)
        state = dict(payload.get("model", payload))
        if args.distill_teacher_use_ema and payload.get("ema"):
            # The shadow tracks trainable params only; buffers stay raw.
            state.update(payload["ema"])
        state = {
            k.removeprefix("flow_lm."): v for k, v in state.items() if k.startswith("flow_lm.")
        }
        load_state_dict_allowing_semantic(
            teacher,
            state,
            strict=True,
            recorded_encoder=payload.get(ENCODER_RECORD_KEY),
            source=str(args.distill_teacher_weights),
        )
        teacher = teacher.eval()
        # Seed the student from the teacher: non-backbone tensors verbatim,
        # backbone from the teacher's bottom+top layers — far faster to
        # converge than a random init.
        seeded, kept = shrink(teacher.state_dict(), len(flow_lm.transformer.layers))
        student_state = flow_lm.state_dict()
        copied = {
            k: v
            for k, v in seeded.items()
            if k in student_state and student_state[k].shape == v.shape
        }
        flow_lm.load_state_dict(copied, strict=False)
        logger.info(
            f"depth distillation: teacher={args.distill_teacher_config}, "
            f"seeded {len(copied)} tensors, kept teacher layers {kept}"
        )
    else:
        assert args.start_from_pretrained, (
            "CFG distillation needs pretrained weights as the teacher"
        )
        teacher = copy.deepcopy(flow_lm).eval()
    disable_grad(teacher)
    stamp_state_names(teacher)
    # nn.Module.__setattr__ would register an nn.Module value as a submodule,
    # pulling the (frozen) teacher into model.parameters()/state_dict() and
    # DDP's broadcast; __dict__ keeps it reachable without that.
    model.__dict__["distill_teacher"] = teacher
    # Only the backbone (and conditioning) learn; the per-frame heads keep
    # the teacher's weights so single-pass generation stays calibrated.
    disable_grad(flow_lm.flow_net)
    disable_grad(flow_lm.out_eos)
    # The flow objective (e.g. LSD's w_s_t weighting net) never runs in
    # distill mode; freeze its params so DDP with
    # find_unused_parameters=False sees no grad-less trainables.
    disable_grad(model.flow)


def build_models(args: TrainArgs) -> tuple[TrainableTTS, MimiModel, tp.Any]:
    """Build (trainable model, frozen mimi, pocket config) from a pocket-tts config."""
    config = load_model_config(args.model_config, args.model_overrides, args.semantic_conditioner)
    tts_model = TTSModel._from_pydantic_config(
        config, temp=0.7, sampler_decode_steps=1, noise_clamp=None, eos_threshold=0.0, origin=None
    )
    flow_lm = tts_model.flow_lm
    d_model = config.flow_lm.transformer.d_model
    latent_dim = config.mimi.inner_dim or config.mimi.seanet.dimension
    flow_lm.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros((d_model, latent_dim), dtype=torch.float32)
    )

    flow = build_flow(args.flow.type, **args.flow.kwargs)
    if flow.num_time_conds != 2:
        # LSD keeps the stock pocket-tts head; other objectives need a different
        # number of time embeddings.
        flow_lm.flow_net = SimpleMLPAdaLN(
            latent_dim,
            config.flow_lm.flow.dim,
            latent_dim,
            d_model,
            config.flow_lm.flow.depth,
            flow.num_time_conds,
        )

    mimi = build_mimi(config.mimi)
    state = None
    if config.weights_path is not None:
        weights_file = download_if_necessary(str(config.weights_path))
        state = safetensors.torch.load_file(weights_file)
        mimi_state = {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}
        mimi.load_state_dict(mimi_state, strict=True)
    else:
        raise ValueError(
            "model_config must define weights_path (used at least for the Mimi codec weights)."
        )

    if args.start_from_pretrained:
        recorded_encoder = read_encoder_record(weights_file)
        flow_state = {
            k.removeprefix("flow_lm."): v for k, v in state.items() if k.startswith("flow_lm.")
        }
        dropped: list[str] = []
        if args.reset_text_embedding:
            dropped = [k for k in flow_state if k.startswith("conditioner.embed.")]
            logger.info("starting the text embedding from scratch: %s", dropped)
            flow_state = {k: v for k, v in flow_state.items() if k not in dropped}
        if flow.num_time_conds != 2:
            flow_state = {k: v for k, v in flow_state.items() if not k.startswith("flow_net.")}
            missing, unexpected = load_state_dict_allowing_semantic(
                flow_lm,
                flow_state,
                strict=False,
                recorded_encoder=recorded_encoder,
                source=str(config.weights_path),
            )
            assert not unexpected, unexpected
            assert all(k.startswith("flow_net.") for k in missing), missing
            logger.info(
                "warm-started backbone; flow_net freshly initialized (objective %s)", args.flow.type
            )
            dit_init(flow_lm.flow_net)
        else:
            # A plain (pre-semantic) checkpoint leaves the fusion at its zero init.
            missing, unexpected = load_state_dict_allowing_semantic(
                flow_lm,
                flow_state,
                strict=not dropped,
                recorded_encoder=recorded_encoder,
                source=str(config.weights_path),
            )
            assert not unexpected, unexpected
            assert set(missing) <= set(dropped), missing
    else:
        gaussian_init(flow_lm.transformer)
        gaussian_init(flow_lm.input_linear)
        gaussian_init(flow_lm.conditioner)
        nn.init.trunc_normal_(flow_lm.speaker_proj_weight, std=1 / math.sqrt(latent_dim))
        dit_init(flow_lm.flow_net)

    mimi.eval()
    stamp_state_names(mimi)
    disable_grad(mimi)
    if flow_lm.semantic_fusion is not None:
        # Fail before training starts; the encoder stays frozen and outside the module tree.
        flow_lm.semantic_fusion.load_encoder()
        logger.info(
            "semantic conditioning on: %s %s@%s layers %d-%d, %s gate",
            config.flow_lm.semantic_conditioner.encoder,
            config.flow_lm.semantic_conditioner.model_name,
            config.flow_lm.semantic_conditioner.revision,
            config.flow_lm.semantic_conditioner.first_layer,
            config.flow_lm.semantic_conditioner.last_layer,
            config.flow_lm.semantic_conditioner.gate,
        )

    model = TrainableTTS(flow_lm, flow, args)
    if args.distill_cfg_coef > 0:
        attach_distillation(model, flow_lm, args)
    return model, mimi, config
