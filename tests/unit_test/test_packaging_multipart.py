"""Regression test: File/Form HTTP routes must register at app startup.

sglang_omni/serve/openai_api.py registers ``POST /v1/audio/voices``
(UploadFile = File(...), Form(...)) unconditionally via ``register_voices``
inside ``create_app`` (openai_api.py:327, :363-369), and
sglang_omni/serve/speech_to_text.py declares further File/Form parameters.
FastAPI raises ``RuntimeError: Form data requires "python-multipart"`` at
route registration when the package is absent, so ``sgl-omni serve`` fails
to start on any platform. python-multipart is therefore declared as a
runtime dependency in every pyproject variant that declares fastapi.
"""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import File, Form, UploadFile  # noqa: E402


def _pyproject_variants_declaring_fastapi(repo_root: Path) -> list[Path]:
    return [
        p
        for p in sorted(repo_root.glob("pyproject*.toml"))
        if "fastapi" in p.read_text()
    ]


def test_form_file_route_registers_at_startup() -> None:
    """Route registration with File/Form params succeeds iff python-multipart is installed."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/v1/audio/voices")
    async def upload_voice(
        audio_sample: UploadFile = File(...),
        consent: str = Form(...),
        name: str = Form(...),
    ) -> dict:
        return {"ok": True}

    paths = {r.path for r in app.routes}
    assert "/v1/audio/voices" in paths


def test_python_multipart_declared_wherever_fastapi_is_declared() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    variants = _pyproject_variants_declaring_fastapi(repo_root)
    assert variants, "expected at least one pyproject variant declaring fastapi"
    missing = [p.name for p in variants if "python-multipart" not in p.read_text()]
    assert not missing, f"python-multipart not declared in: {missing}"
