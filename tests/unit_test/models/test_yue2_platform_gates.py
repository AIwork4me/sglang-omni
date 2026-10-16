# SPDX-License-Identifier: Apache-2.0
"""Dispatch-level tests for the YuE2 platform gates.

These prove which execution branch each platform selects. They mock the
platform and are not GPU execution proof; on-device validation is separate.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.yue2 import fa3 as fa3_module
from sglang_omni.models.yue2 import fastpath, song, stages


class FakePlatform:
    def __init__(self, cuda: bool, rocm: bool):
        self.cuda = cuda
        self.rocm = rocm

    def is_cuda(self) -> bool:
        return self.cuda

    def is_rocm(self) -> bool:
        return self.rocm


class StubSynthesizer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device


@pytest.fixture()
def reset_backend_caches():
    fastpath._IN_TREE = None  # noqa: leading-underscore
    fastpath._PIP = None  # noqa: leading-underscore
    fa3_module._VARLEN = None  # noqa: leading-underscore
    fa3_module._LOADED = False  # noqa: leading-underscore
    yield
    fastpath._IN_TREE = None  # noqa: leading-underscore
    fastpath._PIP = None  # noqa: leading-underscore
    fa3_module._VARLEN = None  # noqa: leading-underscore
    fa3_module._LOADED = False  # noqa: leading-underscore


@pytest.fixture()
def preserve_torch_backend_flags():
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled()
    yield
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp_enabled)


def make_executor(monkeypatch: pytest.MonkeyPatch, platform: FakePlatform):
    monkeypatch.setattr(stages, "current_platform", platform)
    monkeypatch.setattr(stages, "Yue2Synthesizer", StubSynthesizer)
    return stages.create_synth_executor("/tmp/yue2-model")


def test_cuda_platform_accepted(monkeypatch, preserve_torch_backend_flags):
    monkeypatch.delenv("SGLANG_YUE2_BATCH_MAX_SIZE", raising=False)
    executor = make_executor(monkeypatch, FakePlatform(cuda=True, rocm=False))
    assert executor.max_batch_size == 4


def test_rocm_platform_accepted(monkeypatch, preserve_torch_backend_flags):
    monkeypatch.delenv("SGLANG_YUE2_BATCH_MAX_SIZE", raising=False)
    executor = make_executor(monkeypatch, FakePlatform(cuda=False, rocm=True))
    assert executor.max_batch_size == 1


def test_rocm_batch_size_env_override_still_honored(
    monkeypatch, preserve_torch_backend_flags
):
    monkeypatch.setenv("SGLANG_YUE2_BATCH_MAX_SIZE", "3")
    executor = make_executor(monkeypatch, FakePlatform(cuda=False, rocm=True))
    assert executor.max_batch_size == 3


def test_cpu_platform_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match="CUDA or ROCm"):
        make_executor(monkeypatch, FakePlatform(cuda=False, rocm=False))


def run_song(
    monkeypatch: pytest.MonkeyPatch,
    platform: FakePlatform,
    step_graph_result,
    looped_result,
):
    monkeypatch.setattr(song, "current_platform", platform)

    def fake_step_graph(*args, **kwargs):
        return step_graph_result

    def fake_looped(*args, **kwargs):
        return looped_result

    monkeypatch.setattr(song, "generate_song_tokens_step_graph", fake_step_graph)
    monkeypatch.setattr(song, "generate_song_tokens_looped", fake_looped)
    request = SimpleNamespace(cot="full", seed=7, abc=None)
    session = SimpleNamespace(device=torch.device("cuda"))
    return song.generate_song_tokens(
        model=None,
        session=session,
        tokenizer=None,
        request=request,
        abc_sampling=None,
        semantic_sampling=None,
    )


def test_rocm_step_graph_bypassed(monkeypatch):
    monkeypatch.delenv("SGLANG_YUE2_STEP_GRAPH", raising=False)
    result = run_song(
        monkeypatch,
        FakePlatform(cuda=False, rocm=True),
        step_graph_result="step-graph",
        looped_result="looped",
    )
    assert result == "looped"


def test_cuda_step_graph_selected(monkeypatch):
    monkeypatch.delenv("SGLANG_YUE2_STEP_GRAPH", raising=False)
    result = run_song(
        monkeypatch,
        FakePlatform(cuda=True, rocm=False),
        step_graph_result="step-graph",
        looped_result="looped",
    )
    assert result == "step-graph"


def test_step_graph_env_flag_still_forces_looped_on_cuda(monkeypatch):
    monkeypatch.setenv("SGLANG_YUE2_STEP_GRAPH", "0")
    result = run_song(
        monkeypatch,
        FakePlatform(cuda=True, rocm=False),
        step_graph_result="step-graph",
        looped_result="looped",
    )
    assert result == "looped"


def test_rocm_fast_sampler_disabled(reset_backend_caches, monkeypatch):
    monkeypatch.setattr(
        fastpath, "current_platform", FakePlatform(cuda=False, rocm=True)
    )
    monkeypatch.delenv("SGLANG_YUE2_FAST_SAMPLER", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert fastpath.fast_sampler_available() is False


def test_cuda_fast_sampler_unchanged(reset_backend_caches, monkeypatch):
    monkeypatch.setattr(
        fastpath, "current_platform", FakePlatform(cuda=True, rocm=False)
    )
    monkeypatch.delenv("SGLANG_YUE2_FAST_SAMPLER", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert fastpath.fast_sampler_available() is True


def test_fast_sampler_env_disable_still_wins(reset_backend_caches, monkeypatch):
    monkeypatch.setattr(
        fastpath, "current_platform", FakePlatform(cuda=True, rocm=False)
    )
    monkeypatch.setenv("SGLANG_YUE2_FAST_SAMPLER", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert fastpath.fast_sampler_available() is False


def inject_shared_varlen(monkeypatch: pytest.MonkeyPatch, shared_op):
    module = types.ModuleType("sglang.kernels.ops.attention.flash_attention")
    module.flash_attn_varlen_func = shared_op
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.ops.attention.flash_attention", module
    )


def test_rocm_varlen_uses_native_aten(reset_backend_caches, monkeypatch):
    monkeypatch.setattr(
        fa3_module, "current_platform", FakePlatform(cuda=False, rocm=True)
    )
    monkeypatch.setenv("SGLANG_YUE2_FA3", "1")
    inject_shared_varlen(
        monkeypatch,
        shared_op=lambda *args, **kwargs: pytest.fail(
            "shared FA op must not run on ROCm"
        ),
    )
    assert fa3_module.fa3_available() is False

    calls = []

    def fake_aten_forward(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(1)

    monkeypatch.setattr(torch.ops.aten, "_flash_attention_forward", fake_aten_forward)
    q = k = v = torch.zeros(2, 1, 8)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 2], dtype=torch.int32)
    out = fa3_module.varlen_attention(q, k, v, cu, cu, 2, 2, seqused_k=seqused_k)
    assert out == torch.zeros(1)
    assert len(calls) == 1
    assert calls[0][1]["seqused_k"] is seqused_k


def test_cuda_varlen_dispatch_unchanged(reset_backend_caches, monkeypatch):
    monkeypatch.setattr(
        fa3_module, "current_platform", FakePlatform(cuda=True, rocm=False)
    )
    monkeypatch.setenv("SGLANG_YUE2_FA3", "1")
    calls = []

    def shared_op(*args, **kwargs):
        calls.append((args, kwargs))
        return "shared-op-output"

    inject_shared_varlen(monkeypatch, shared_op=shared_op)
    assert fa3_module.fa3_available() is True

    q = k = v = torch.zeros(2, 1, 8)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 2], dtype=torch.int32)
    out = fa3_module.varlen_attention(q, k, v, cu, cu, 2, 2, seqused_k=seqused_k)
    assert out == "shared-op-output"
    assert len(calls) == 1
    assert calls[0][1] == {"seqused_k": seqused_k, "causal": False, "ver": 3}
