"""Model registry: detection, support status, and loading.

Maps Hugging Face model ids/architectures to adapters, tracks the support
status used in the README table, and gives clear errors for unsupported or
quantized checkpoints (quantized Jacobian fitting is experimental future work,
not part of v0.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from openjspace.config import resolve_device, resolve_dtype
from openjspace.models.hf_decoder import HFDecoderAdapter, UnsupportedArchitectureError
from openjspace.models.protocol import LensModelAdapter

ModelKind = Literal["text", "vlm"]
SupportStatus = Literal["tested", "experimental", "planned", "unsupported"]


@dataclass(frozen=True)
class ModelFamily:
    name: str
    kind: ModelKind
    status: SupportStatus
    architectures: tuple[str, ...]
    example_id: str
    notes: str = ""


#: Support table surfaced by ``openjspace models list`` and the README.
#: A family is only marked "tested" once an integration test has passed.
MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="Qwen2 / Qwen2.5 (text)",
        kind="text",
        status="tested",
        architectures=("Qwen2ForCausalLM",),
        example_id="Qwen/Qwen2.5-0.5B-Instruct",
        notes="Primary development family; integration-tested on Qwen2.5-0.5B-Instruct",
    ),
    ModelFamily(
        name="Qwen3 (text)",
        kind="text",
        status="experimental",
        architectures=("Qwen3ForCausalLM",),
        example_id="Qwen/Qwen3-0.6B",
    ),
    ModelFamily(
        name="Llama 3.x",
        kind="text",
        status="experimental",
        architectures=("LlamaForCausalLM",),
        example_id="meta-llama/Llama-3.2-1B",
        notes="Same layout as Qwen; no integration test yet",
    ),
    ModelFamily(
        name="Mistral",
        kind="text",
        status="experimental",
        architectures=("MistralForCausalLM",),
        example_id="mistralai/Mistral-7B-v0.3",
    ),
    ModelFamily(
        name="Gemma 2/3 (text)",
        kind="text",
        status="experimental",
        architectures=("Gemma2ForCausalLM", "Gemma3ForCausalLM"),
        example_id="google/gemma-2-2b",
        notes="Final logit softcapping handled",
    ),
    ModelFamily(
        name="GPT-2",
        kind="text",
        status="experimental",
        architectures=("GPT2LMHeadModel",),
        example_id="openai-community/gpt2",
    ),
    ModelFamily(
        name="Pythia / GPT-NeoX",
        kind="text",
        status="experimental",
        architectures=("GPTNeoXForCausalLM",),
        example_id="EleutherAI/pythia-160m",
    ),
    ModelFamily(
        name="SmolVLM / SmolVLM2",
        kind="vlm",
        status="tested",
        architectures=("SmolVLMForConditionalGeneration", "Idefics3ForConditionalGeneration"),
        example_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        notes="Pixel-shuffle connector: patch mapping is approximate; "
        "integration-tested on SmolVLM-256M-Instruct",
    ),
    ModelFamily(
        name="Qwen2.5-VL / Qwen3-VL",
        kind="vlm",
        status="planned",
        architectures=("Qwen2_5_VLForConditionalGeneration",),
        example_id="Qwen/Qwen2.5-VL-3B-Instruct",
        notes="Planned richer VLM adapter",
    ),
)


def family_for_architecture(architecture: str) -> ModelFamily | None:
    for family in MODEL_FAMILIES:
        if architecture in family.architectures:
            return family
    return None


def detect_model_kind(config: object) -> ModelKind:
    """Text vs. VLM detection from an HF config object."""
    if getattr(config, "vision_config", None) is not None:
        return "vlm"
    architectures = getattr(config, "architectures", None) or []
    if any("ConditionalGeneration" in a or "VL" in a for a in architectures):
        return "vlm"
    return "text"


def _reject_quantized(config: object, model_id: str) -> None:
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError(
            f"{model_id} is a quantized checkpoint. Quantized Jacobian fitting is "
            "not supported in v0.1 (gradients through dequantization kernels are "
            "unreliable); use an unquantized checkpoint. Quantized support is "
            "documented as experimental future work."
        )


@dataclass
class LoadedModel:
    adapter: LensModelAdapter
    kind: ModelKind
    device: str
    dtype: str


def load_model(
    model_id: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load a Hugging Face model and wrap it in the right adapter.

    Args:
        model_id: HF repo id or local path (``openjspace/tiny-test-model``
            loads the built-in tiny CPU model).
        device: ``auto``/``cuda``/``mps``/``cpu``.
        dtype: ``auto`` (bf16 on capable CUDA, fp16 on MPS, fp32 on CPU) or an
            explicit dtype string. Lens accumulation is always float32.
        revision: Optional HF revision (branch/tag/commit).
        trust_remote_code: Off by default; arbitrary remote-code models are
            out of scope for v0.1 and require explicit opt-in.

    Raises:
        ValueError: For quantized checkpoints or unavailable devices.
        UnsupportedArchitectureError: When the internal layout is unknown.
    """
    from openjspace.models.tiny import TINY_MODEL_ID, TinyAdapter

    if model_id == TINY_MODEL_ID:
        return LoadedModel(adapter=TinyAdapter(), kind="text", device="cpu", dtype="float32")

    from transformers import AutoConfig, AutoTokenizer

    resolved_device = resolve_device(device)
    config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    _reject_quantized(config, model_id)
    kind = detect_model_kind(config)
    torch_dtype = resolve_dtype(dtype, resolved_device)

    adapter: LensModelAdapter
    if kind == "vlm":
        from openjspace.models.smolvlm import load_smolvlm_adapter

        adapter = load_smolvlm_adapter(
            model_id,
            config=config,
            device=resolved_device,
            torch_dtype=torch_dtype,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        return LoadedModel(
            adapter=adapter, kind="vlm", device=resolved_device, dtype=str(torch_dtype)
        )

    from transformers import AutoModelForCausalLM

    hf_model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ),
    )
    hf_model.to(resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    architectures = tuple(getattr(config, "architectures", None) or [])
    if any(a in ("Qwen2ForCausalLM", "Qwen3ForCausalLM") for a in architectures):
        from openjspace.models.qwen_decoder import QwenDecoderAdapter

        adapter = QwenDecoderAdapter(
            hf_model, tokenizer, model_id=model_id, model_revision=revision
        )
    else:
        try:
            adapter = HFDecoderAdapter(
                hf_model, tokenizer, model_id=model_id, model_revision=revision
            )
        except UnsupportedArchitectureError:
            raise
    return LoadedModel(adapter=adapter, kind="text", device=resolved_device, dtype=str(torch_dtype))
