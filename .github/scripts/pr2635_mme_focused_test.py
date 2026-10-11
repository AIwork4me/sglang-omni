"""Capture the remaining Video-MME answer under isolated request histories."""

import asyncio
import os
from pathlib import Path

from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from tests.test_model.conftest import OMNI_CI_QWEN_FIXTURES
from tests.test_model.omni_ci_config import OmniCiModelPreset
from tests.test_model.omni_router_utils import ManagedRouterHandle
from tests.test_model.test_qwen3_omni_videomme_talker_ci import (
    load_short_answer_samples,
)

OMNI_CI_QWEN_FIXTURES["test_pr2635_mme_talker_focused"] = (
    "qwen3_omni_bf16_disagg_server"
)


def test_mme_talker_focused(
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ManagedRouterHandle,
) -> None:
    sample_ids = (
        {"007-1", "007-2"} if os.environ["PR2635_HISTORY"] == "pair" else {"007-2"}
    )
    samples = [
        sample
        for sample in load_short_answer_samples()
        if sample.sample_id in sample_ids
    ]
    config = VideoEvalConfig(
        model=omni_ci_model.name,
        port=omni_ci_server.port,
        max_samples=len(samples),
        max_tokens=256,
        max_concurrency=1,
        output_dir=str(Path(os.environ["PR2635_OUTPUT"])),
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        seed=123,
        warmup=0,
        temperature=0,
        disable_tqdm=True,
        timeout_s=500,
    )
    results = asyncio.run(
        run_video_eval(
            config,
            samples=samples,
            task_label="Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default="results/videomme_audio",
            compute_wer=False,
        )
    )
    assert results["summary"]["failed"] == 0
