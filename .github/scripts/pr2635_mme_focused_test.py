"""Capture the remaining Video-MME answer under isolated request histories."""

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from tests.test_model.omni_ci_config import OmniCiModelPreset
from tests.test_model.omni_router_utils import (
    CiRouterTopology,
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.test_model.test_qwen3_omni_videomme_talker_ci import (
    load_short_answer_samples,
)


@pytest.fixture(scope="module")
def omni_ci_server(
    tmp_path_factory: pytest.TempPathFactory,
    omni_ci_model: OmniCiModelPreset,
) -> Iterator[ManagedRouterHandle]:
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=os.environ["SGLANG_OMNI_TEST_MINICPMO_MODEL"],
        model_name=omni_ci_model.name,
        worker_extra_args=(
            "--thinker.factory.max_seq_len 32768 "
            "--thinker.engine.mem_fraction_static 0.55 "
            "--talker.engine.mem_fraction_static 0.15"
        ),
        router_topology=CiRouterTopology.OMNI_AUDIO,
        num_workers=1,
        num_gpus_per_worker=1,
        generation_streaming=False,
    ) as router:
        yield router


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
