"""Startup model-loading contract: primary paths, fallback cache, and failure modes."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import joblib
import pytest
from asgi_lifespan import LifespanManager

from app.main import _write_fallback_cache


def _make_mock_engine() -> MagicMock:
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    return mock_engine


def _exists_factory(present: set[str]):
    """Return an os.path.exists replacement that reports only the named files."""
    def _exists(path):
        return any(name in str(path) for name in present)
    return _exists


# ── _write_fallback_cache ────────────────────────────────────────────────


def test_write_fallback_cache_writes_atomically(tmp_path) -> None:
    cache = tmp_path / "fallback.pkl"
    _write_fallback_cache({"weights": [1, 2, 3]}, str(cache))

    assert joblib.load(cache) == {"weights": [1, 2, 3]}
    assert not (tmp_path / "fallback.pkl.tmp").exists()


def test_write_fallback_cache_failure_is_non_fatal_and_leaves_no_partial_file(tmp_path) -> None:
    cache = tmp_path / "fallback.pkl"
    unpicklable = lambda x: x  # noqa: E731 - lambdas cannot be pickled

    _write_fallback_cache(unpicklable, str(cache))  # must not raise

    assert not cache.exists()
    assert not (tmp_path / "fallback.pkl.tmp").exists()


def test_write_fallback_cache_failure_keeps_previous_cache(tmp_path) -> None:
    cache = tmp_path / "fallback.pkl"
    joblib.dump("previous-good-model", cache)

    _write_fallback_cache(lambda x: x, str(cache))

    assert joblib.load(cache) == "previous-good-model"


# ── lifespan model loading ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lifespan_prefers_calibrated_model_when_present() -> None:
    calibrated = MagicMock(name="CalibratedModel")
    scaler = MagicMock(name="Scaler")

    def mock_load(path):
        if "calibrated_model.pkl" in str(path):
            return calibrated
        if "preprocessor.pkl" in str(path):
            return scaler
        raise ValueError(f"Unexpected joblib load for path: {path}")

    with (
        patch("app.main.validate_model_manifest", return_value=None),
        patch("os.path.exists", side_effect=_exists_factory({"calibrated_model.pkl"})),
        patch("mlflow.lightgbm.load_model") as mlflow_load,
        patch("joblib.load", side_effect=mock_load),
        patch("app.main._write_fallback_cache") as write_cache,
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            assert app.state.model is calibrated

    mlflow_load.assert_not_called()
    write_cache.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_keeps_loaded_model_when_cache_write_fails() -> None:
    """Regression: a failed cache refresh must not discard a successfully loaded model."""
    loaded = MagicMock(name="MlflowModel")

    with (
        patch("app.main.validate_model_manifest", return_value=None),
        patch("os.path.exists", side_effect=_exists_factory({"preprocessor.pkl"})),
        patch("mlflow.lightgbm.load_model", return_value=loaded),
        patch("joblib.load", return_value=MagicMock(name="Scaler")),
        patch("joblib.dump", side_effect=OSError("disk full")),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            assert app.state.model is loaded


@pytest.mark.asyncio
async def test_lifespan_aborts_when_manifest_validation_fails() -> None:
    with (
        patch("app.main.validate_model_manifest", side_effect=ValueError("hash mismatch")),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        with pytest.raises(ValueError, match="hash mismatch"):
            async with LifespanManager(app):
                pass


@pytest.mark.asyncio
async def test_lifespan_aborts_when_model_and_cache_are_both_unavailable() -> None:
    with (
        patch("app.main.validate_model_manifest", return_value=None),
        patch("os.path.exists", side_effect=_exists_factory(set())),
        patch("mlflow.lightgbm.load_model", side_effect=RuntimeError("MLflow down")),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="MLflow down"):
            async with LifespanManager(app):
                pass


@pytest.mark.asyncio
async def test_lifespan_aborts_when_fallback_cache_is_corrupt() -> None:
    with (
        patch("app.main.validate_model_manifest", return_value=None),
        patch("os.path.exists", side_effect=_exists_factory({"calibrated_model_fallback.pkl"})),
        patch("mlflow.lightgbm.load_model", side_effect=RuntimeError("MLflow down")),
        patch("joblib.load", side_effect=EOFError("truncated cache")),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        with pytest.raises(EOFError, match="truncated cache"):
            async with LifespanManager(app):
                pass


def test_fallback_cache_dir_is_git_ignored() -> None:
    """The cache is a runtime artefact and must never be committed."""
    with open(os.path.join(os.path.dirname(__file__), "..", "..", ".gitignore")) as f:
        assert "models/model_cache/" in f.read()
