"""Configuration models for loading YAML config files."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from pocket_tts.utils.utils import download_if_necessary

CONFIGS_DIR = Path(__file__).parent.parent / "config"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    dim: int
    depth: int
    # "lsd" (2 time conditions, 1-step decode) or "flow_matching" (1 time
    # condition, Euler integration; needs >= 16 decode steps).
    type: str = "lsd"


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int


class LookupTable(StrictModel):
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str


DICTABERT_LARGE = "dicta-il/dictabert-large"
DICTABERT_LARGE_REVISION = "a972ffb612c03bcfb128f10f5ef4462f7adb9844"


class SemanticConditionerConfig(StrictModel):
    """Optional DictaBERT features fused into the text prefix (pocket_tts/conditioners/semantic.py).

    Every field has a default, so `semantic_conditioner: {}` turns the feature on as designed.
    """

    model_name: str = DICTABERT_LARGE
    revision: str = DICTABERT_LARGE_REVISION
    # Width of the encoder's hidden states; checked against the encoder when it loads.
    hidden_size: int = 1024
    # Encoder layers mixed by learned softmax weights, 1-based and inclusive (0 would be the
    # embedding output, which is never used).
    first_layer: int = 8
    last_layer: int = 20
    # "scalar": one gate value per token. "vector": one per model channel.
    gate: Literal["scalar", "vector"] = "scalar"
    gate_hidden: int = 256
    # The encoder's position limit, [CLS] and [SEP] included. Longer texts are encoded in
    # independent windows of max_tokens - 2 tokens.
    max_tokens: int = 512

    @model_validator(mode="after")
    def _check_band(self) -> "SemanticConditionerConfig":
        if not 1 <= self.first_layer <= self.last_layer:
            raise ValueError(
                f"semantic_conditioner layers must satisfy 1 <= first_layer <= last_layer, "
                f"got {self.first_layer}..{self.last_layer}"
            )
        if self.max_tokens < 3:
            raise ValueError("semantic_conditioner.max_tokens must leave room for [CLS] and [SEP]")
        return self


# Root configuration
class FlowLMConfig(StrictModel):
    """Root configuration model for YAML config files."""

    dtype: str

    # Nested configurations
    flow: FlowConfig
    transformer: FlowLMTransformerConfig

    # conditioning
    lookup_table: LookupTable
    weights_path: str | None = None
    insert_bos_before_voice: bool = False
    # Off unless present. See SemanticConditionerConfig.
    semantic_conditioner: SemanticConditionerConfig | None = None


# SEANet configuration
class SEANetConfig(StrictModel):
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    dimension: int
    output_dimension: int


# Root configuration
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    # Sample rate and channels
    sample_rate: int
    channels: int
    frame_rate: float

    # SEANet configurations
    seanet: SEANetConfig

    # Transformer
    transformer: MimiTransformerConfig

    # Quantizer
    quantizer: QuantizerConfig
    weights_path: str | None = None
    inner_dim: int | None = None
    outer_dim: int | None = None


class Config(StrictModel):
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    weights_path_without_voice_cloning: str | None = None
    pad_with_spaces_for_short_inputs: bool = False
    remove_semicolons: bool = False
    # Map curly quotes, Unicode dashes and invisible marks to the plain forms the training text
    # was normalized to (pocket_tts/utils/text_fold.py). On for models trained on such text.
    fold_text: bool = False
    model_recommended_frames_after_eos: int | None = None
    default_temperature: float = 0.7


def load_config(yaml_path: str | Path) -> Config:
    yaml_path = download_if_necessary(str(yaml_path))

    if not yaml_path.exists():
        if yaml_path.is_relative_to(CONFIGS_DIR):
            raise FileNotFoundError(
                f"Config file not found: {yaml_path}. "
                f"Did you make a typo? Available languages: {[p.stem for p in CONFIGS_DIR.glob('*.yaml')]}"
            )
        raise FileNotFoundError(f"Config file not found: {yaml_path}. Did you make a typo?")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
