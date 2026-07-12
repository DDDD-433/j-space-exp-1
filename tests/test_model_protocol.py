"""Adapter protocol tests: tiny adapter and mock-HF layout detection."""

import pytest
import torch
from torch import nn

from openjspace.models.hf_decoder import (
    LAYOUTS,
    HFDecoderAdapter,
    Layout,
    UnsupportedArchitectureError,
    find_layout,
)
from openjspace.models.protocol import LensModelAdapter
from openjspace.models.tiny import TinyAdapter


class _MockConfig:
    def __init__(self, n_layers: int, hidden: int) -> None:
        self.num_hidden_layers = n_layers
        self.hidden_size = hidden
        self._name_or_path = "mock/model"

    def get_text_config(self):
        return self


class _MockTokenizer:
    bos_token_id = None
    name_or_path = "mock/tokenizer"
    all_special_ids = [0]

    def __call__(self, text, *, return_tensors="pt", truncation=True, max_length=128):
        from types import SimpleNamespace

        ids = [0, *[1 + b % 90 for b in text.encode()][: max_length - 1]]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, **_kw):
        return "".join(chr(32 + int(i)) for i in ids)


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + 0.1 * self.linear(hidden_states),)


def _make_hf_mock(layout: Layout, *, n_layers=3, hidden=8, vocab=100) -> nn.Module:
    class _TextModule(nn.Module):
        def forward(self, input_ids=None, use_cache=False, **kwargs):
            hidden_states = getattr(self, layout.embed)(input_ids)
            for block in getattr(self, layout.layers):
                (hidden_states,) = block(hidden_states)
            return hidden_states

    text = _TextModule()
    setattr(text, layout.layers, nn.ModuleList(_Block(hidden) for _ in range(n_layers)))
    setattr(text, layout.norm, nn.LayerNorm(hidden))
    setattr(text, layout.embed, nn.Embedding(vocab, hidden))

    root = nn.Module()
    parent = root
    *parts, leaf = layout.path.split(".")
    for part in parts:
        child = nn.Module()
        setattr(parent, part, child)
        parent = child
    setattr(parent, leaf, text)
    setattr(root, layout.lm_head, nn.Linear(hidden, vocab, bias=False))
    root.config = _MockConfig(n_layers, hidden)
    return root


@pytest.mark.parametrize(
    "layout",
    [
        Layout("model"),
        Layout("model.language_model"),
        Layout("model", norm="final_layernorm"),
        Layout("transformer", layers="h", norm="ln_f", embed="wte"),
        Layout("gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"),
    ],
    ids=["llama", "multimodal", "phi", "gpt2", "gptneox"],
)
def test_find_layout_roundtrip(layout):
    assert find_layout(_make_hf_mock(layout)) == layout


def test_layout_table_nonempty():
    assert len(LAYOUTS) >= 5


def test_unsupported_architecture_actionable_error():
    bad = nn.Module()
    bad.something = nn.Module()
    with pytest.raises(UnsupportedArchitectureError, match="could not locate"):
        find_layout(bad)


def test_hf_adapter_structure_over_mock():
    adapter = HFDecoderAdapter(
        _make_hf_mock(Layout("model")), _MockTokenizer(), model_id="mock/model"
    )
    assert isinstance(adapter, LensModelAdapter)
    assert adapter.n_layers == 3
    assert adapter.hidden_size == 8
    assert adapter.vocab_size == 100
    assert adapter.residual_location == "block_output"
    assert adapter.get_unembedding_weight().shape == (100, 8)
    assert len(adapter.get_residual_modules()) == 3
    logits = adapter.unembed(torch.randn(1, 5, 8))
    assert logits.shape == (1, 5, 100)


def test_hf_adapter_hooks_return_differentiable_activations():
    adapter = HFDecoderAdapter(
        _make_hf_mock(Layout("model")), _MockTokenizer(), model_id="mock/model"
    )
    inputs = adapter.prepare_inputs("hello world, this is a test prompt")
    with torch.enable_grad():
        result = adapter.forward_with_activations(inputs, layers=[0, 2], grad_from=0)
        assert result.activations[0].requires_grad
        assert result.activations[2].requires_grad
        grad = torch.autograd.grad(
            result.activations[2].sum(), result.activations[0], allow_unused=False
        )[0]
    assert grad is not None and torch.isfinite(grad).all()


def test_hf_adapter_inference_mode_when_no_grad_requested():
    adapter = HFDecoderAdapter(
        _make_hf_mock(Layout("model")), _MockTokenizer(), model_id="mock/model"
    )
    inputs = adapter.prepare_inputs("hello world, this is a test prompt")
    result = adapter.forward_with_activations(inputs, layers=[1])
    assert not result.activations[1].requires_grad


def test_hf_adapter_rejects_images():
    adapter = HFDecoderAdapter(
        _make_hf_mock(Layout("model")), _MockTokenizer(), model_id="mock/model"
    )
    with pytest.raises(ValueError, match="text-only"):
        adapter.prepare_inputs("hi", images=[object()])


def test_hf_adapter_layer_count_mismatch_rejected():
    mock = _make_hf_mock(Layout("model"), n_layers=3)
    mock.config.num_hidden_layers = 5
    with pytest.raises(UnsupportedArchitectureError, match="num_hidden_layers"):
        HFDecoderAdapter(mock, _MockTokenizer(), model_id="mock/model")


def test_tiny_adapter_protocol_surface():
    adapter = TinyAdapter()
    assert isinstance(adapter, LensModelAdapter)
    assert adapter.n_layers == 4
    assert adapter.hidden_size == 8
    assert adapter.vocab_size == 32
    assert adapter.get_unembedding_weight().shape == (32, 8)
    inputs = adapter.prepare_inputs("hello")
    positions = adapter.classify_positions(inputs)
    assert positions[0].modality == "special"  # BOS
    assert all(p.modality == "text" for p in positions[1:])
    decoded = adapter.decode_token_ids(inputs.input_ids[0].tolist())
    assert len(decoded) == inputs.seq_len


def test_registry_quantized_rejected():
    from openjspace.models.registry import _reject_quantized

    class _Cfg:
        quantization_config = {"bits": 4}

    with pytest.raises(ValueError, match="quantized"):
        _reject_quantized(_Cfg(), "some/quantized-model")


def test_registry_kind_detection():
    from openjspace.models.registry import detect_model_kind

    class _Text:
        vision_config = None
        architectures = ["Qwen2ForCausalLM"]

    class _VLM:
        vision_config = object()
        architectures = ["SmolVLMForConditionalGeneration"]

    assert detect_model_kind(_Text()) == "text"
    assert detect_model_kind(_VLM()) == "vlm"


def test_registry_support_table():
    from openjspace.models.registry import MODEL_FAMILIES, family_for_architecture

    assert family_for_architecture("Qwen2ForCausalLM") is not None
    assert family_for_architecture("TotallyMadeUpModel") is None
    statuses = {f.status for f in MODEL_FAMILIES}
    assert statuses <= {"tested", "experimental", "planned", "unsupported"}
