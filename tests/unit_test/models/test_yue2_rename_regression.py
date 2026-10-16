# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the YuE2 public-naming closure (omni lint policy).

The leading-underscore rename is required by the repo-wide naming policy
(#2267/#2338/#2365) and must stay behavior-preserving: these tests pin the
new public names, prove the old names are gone at the module/class surface,
and check the invariants the rename must not disturb (state_dict stability,
config serialization, fused-weight cache attributes, lease accounting).
"""
from __future__ import annotations

import inspect

import torch

from sglang_omni.models.yue2 import (
    batched,
    cuda_graph,
    fa3,
    fastpath,
    modeling_vae,
    nar,
    nar_fast,
    sampling,
    song,
)


def public(module_or_class, name: str) -> bool:
    return hasattr(module_or_class, name)


def test_cross_file_lazy_loaders_are_public():
    # song.py imports these from fastpath/sampling; a missed edge here is an
    # ImportError at request time, not import time.
    from sglang_omni.models.yue2.fastpath import load_topk_backend
    from sglang_omni.models.yue2.sampling import decode_loop

    assert callable(load_topk_backend)
    assert callable(decode_loop)
    assert not hasattr(fastpath, "_load_topk_backend")
    assert not hasattr(sampling, "_decode_loop")
    assert callable(song.load_topk_backend)
    assert callable(song.decode_loop)


def test_fastpath_backend_loader_and_sampler_surface():
    from sglang_omni.models.yue2.fastpath import TorchTopKBackend

    assert callable(TorchTopKBackend.topk)
    assert not hasattr(fastpath, "_TorchTopKBackend")
    backend = fastpath.load_topk_backend()
    assert backend is not None and backend is not False
    assert callable(fastpath.FastSampler.phase_mask)
    assert callable(fastpath.BatchedGraphSampler.fill_scalars)
    assert callable(fastpath.window_penalty_inplace)
    assert not hasattr(fastpath.FastSampler, "_phase_mask")
    assert not hasattr(fastpath.BatchedGraphSampler, "_fill_scalars")
    assert not hasattr(fastpath, "_window_penalty_")


def test_cuda_graph_class_surface():
    for name in ("decode", "decode_fused", "step_once", "capture_graph"):
        assert callable(getattr(cuda_graph.GraphAR, name)), name
    for name in ("_decode", "_decode_fused", "_step_once", "_capture"):
        assert not hasattr(cuda_graph.GraphAR, name), name
    assert inspect.isclass(cuda_graph.PrefixCache)
    assert not hasattr(cuda_graph, "_PrefixCache")
    assert callable(cuda_graph.fused_enabled)
    assert not hasattr(cuda_graph, "_fused_enabled")
    # The bool attribute and the renamed capture method must coexist.
    assert "capture" in inspect.getsource(cuda_graph.GraphAR.__init__)


def test_session_pool_attributes_are_public():
    pool = cuda_graph.GraphSessionPool()
    assert pool.idle == {}
    assert pool.live == 0
    assert hasattr(pool.lock, "acquire")
    assert not hasattr(pool, "_idle") and not hasattr(pool, "_live")
    assert not hasattr(pool, "_lock")


def test_modeling_helpers_public_and_rotary_state_dict_stable():
    from sglang_omni.models.yue2 import modeling_yue2

    assert callable(modeling_yue2.build_causal_mask)
    assert callable(modeling_yue2.apply_rotary)
    assert not hasattr(modeling_yue2, "_causal_mask")
    assert not hasattr(modeling_yue2, "_apply_rotary")
    assert callable(modeling_yue2.YuE2ForCausalLM.shift_t_value)
    assert not hasattr(modeling_yue2.YuE2ForCausalLM, "_shift_t_value")
    # transformers init hook stays under its framework-mandated name.
    assert callable(
        modeling_yue2.YuE2PreTrainedModel._init_weights
    )  # noqa: leading-underscore  # transformers hook
    rotary = modeling_yue2.RotaryEmbedding(8)
    angles = rotary(torch.arange(4)[None])
    assert angles[0].shape == (1, 4, 4)
    assert list(rotary.state_dict().keys()) == []
    assert rotary.inv_freq is not None


def test_config_to_dict_unchanged_by_field_renames():
    from sglang_omni.models.yue2.modeling_yue2 import YuE2Config

    config = YuE2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=16,
        vocab_size=128,
    )
    serialized = config.to_dict()
    assert "hf_fields" not in serialized and "_hf_fields" not in serialized
    assert "inference_fields" not in serialized
    assert serialized["hidden_size"] == 32
    assert serialized["vocab_size"] == 128
    assert serialized["model_type"] == "yue2"


def test_vae_config_to_dict_unchanged_by_field_renames():
    from sglang_omni.models.yue2.modeling_vae import YuE2VAEConfig

    serialized = YuE2VAEConfig().to_dict()
    assert "hf_fields" not in serialized and "inference_fields" not in serialized
    assert serialized["sample_rate"] == 48000
    assert serialized["latent_dim"] == 64


def test_vae_public_helpers():
    assert callable(modeling_vae.dependency_interval)
    assert callable(modeling_vae.output_length)
    assert not hasattr(modeling_vae, "_dependency_interval")
    assert not hasattr(modeling_vae, "_output_length")
    assert callable(modeling_vae.YuE2VAE.validate_latent)
    assert not hasattr(modeling_vae.YuE2VAE, "_latent")


def test_nar_public_surface():
    assert callable(nar.coerce_integers)
    assert not hasattr(nar, "_integers")
    assert callable(nar.attention)
    assert callable(nar.CachedNAR.attention)
    assert callable(nar.CachedNAR.prefill)
    assert not hasattr(nar.CachedNAR, "_attention")
    assert not hasattr(nar.CachedNAR, "_prefill")
    assert callable(nar.offload_ar_modules)
    assert not hasattr(nar, "_offload_ar")
    assert [int(v) for v in nar.coerce_integers([1, 2, 3], "x")] == [1, 2, 3]


def test_nar_fast_public_surface():
    assert callable(nar_fast.load_sgl_kernel)
    assert callable(nar_fast.flash_varlen)
    assert not hasattr(nar_fast, "_sgl_kernel")
    assert not hasattr(nar_fast, "_flash_varlen")
    assert inspect.isclass(nar_fast.VelocityGraph)
    assert not hasattr(nar_fast, "_VelocityGraph")
    pool = nar_fast.VelocityGraphPool()
    assert pool.graphs == {}
    assert hasattr(pool.lock, "acquire")
    assert not hasattr(pool, "_graphs") and not hasattr(pool, "_lock")


def test_song_generator_surface():
    assert callable(song.generate_song_tokens_step_graph)
    assert callable(song.generate_song_tokens_looped)
    assert not hasattr(song, "_generate_song_tokens_step_graph")
    assert not hasattr(song, "_generate_song_tokens_looped")
    assert inspect.isclass(song.StepGraphUnavailable)
    assert not hasattr(song, "_StepGraphUnavailable")
    assert callable(song.song_token_budget)


def test_batched_lease_and_sampler_budget_surface():
    assert callable(batched.abc_per_request)
    assert not hasattr(batched, "_abc_per_request")
    calls = []

    class StubPool:
        def release(self, session):
            calls.append(session)

    session = object()
    lease = batched.BatchedSessionLease(session, refs=2, pool=StubPool())
    assert lease.refs == 2 and lease.released is False
    lease.release()
    assert lease.refs == 1 and calls == []
    lease.release()
    assert lease.refs == 0 and lease.released is True
    assert calls == [session]
    assert not hasattr(lease, "_refs") and not hasattr(lease, "_released")


def test_model_fused_weight_cache_attributes():
    class Proj:
        def __init__(self, dim):
            self.weight = torch.zeros(dim, 4)

    class Attention:
        def __init__(self):
            self.q_proj = Proj(3)
            self.k_proj = Proj(3)
            self.v_proj = Proj(3)

    class MLP:
        def __init__(self):
            self.gate_proj = Proj(4)
            self.up_proj = Proj(4)

    class Layer:
        def __init__(self):
            self.self_attn = Attention()
            self.mlp = MLP()
            self.nar_self_attn = Attention()
            self.nar_mlp = MLP()

    class Model:
        def __init__(self):
            self.layers = [Layer(), Layer()]

    class Root:
        def __init__(self):
            self.model = Model()

    ar_root = Root()
    fused = cuda_graph.model_fused_weights(ar_root)
    assert fused is ar_root.yue2_fused_weights
    assert not hasattr(ar_root, "_yue2_fused_weights")
    assert cuda_graph.model_fused_weights(ar_root) is fused

    nar_root = Root()
    nar_fused = nar_fast.model_fused_nar_weights(nar_root)
    assert nar_fused is nar_root.yue2_fused_nar_weights
    assert not hasattr(nar_root, "_yue2_fused_nar_weights")


def test_fa3_loader_public():
    assert callable(fa3.load_varlen)
    assert not hasattr(fa3, "_load_varlen")
    # Without the shared sglang op the loader returns None (reference path).
    fa3._LOADED = False  # noqa: leading-underscore  # reset lazy cache
    try:
        result = fa3.load_varlen()
    finally:
        fa3._LOADED = False  # noqa: leading-underscore  # reset lazy cache
    assert result is None or callable(result)
