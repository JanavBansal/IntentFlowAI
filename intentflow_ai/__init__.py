"""Top-level package for IntentFlow AI.

This module exposes convenience imports so user code can access key
subsystems (config, data, features, modeling) without deep paths.
"""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - executed only in installed distributions
    __version__ = version("intentflow-ai")
except PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.0.0-dev"

from intentflow_ai.config.settings import Settings
from intentflow_ai.pipelines.training import TrainingPipeline

__all__ = ["__version__", "Settings", "TrainingPipeline"]
