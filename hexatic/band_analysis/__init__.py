"""Stable dilute-band area extraction and inference."""

from .components import BandComponent, label_dilute_bands
from .bayesian import (
    BayesianResult,
    MCMCConfig,
    MCMCDiagnostics,
    coupled_area_model,
    run_bayesian_inference,
)
from .characterization import DetectedBand, characterize_band
from .density import SurfaceGrid
from .inference import (
    HessianDiagnostics,
    OptimizationResult,
    OptimizationRun,
    empirical_parameters,
    optimize_parameters,
)
from .model import (
    PARAMETER_NAMES,
    JITTER,
    Scaling,
    TrainingTransitions,
    TransitionBlock,
    build_transition_blocks,
    negative_log_likelihood,
    positive_parameters,
    prepare_training_transitions,
    raw_parameters,
    transition,
    transition_log_density,
)
from .segments import StableSegment, build_stable_segments
from .tracking import (
    BandTracker,
    DetectionFrame,
    EventCode,
    EventRecord,
    TrackedBand,
    TrackedFrame,
)
from .validation import ValidationResult, validate_posterior
from .workflow import AnalysisConfig, LagOutcome, run_analysis, stable_lags

__all__ = [
    "BandComponent",
    "BandTracker",
    "BayesianResult",
    "DetectedBand",
    "DetectionFrame",
    "EventCode",
    "EventRecord",
    "HessianDiagnostics",
    "JITTER",
    "LagOutcome",
    "MCMCConfig",
    "MCMCDiagnostics",
    "OptimizationResult",
    "OptimizationRun",
    "PARAMETER_NAMES",
    "Scaling",
    "SurfaceGrid",
    "AnalysisConfig",
    "TrainingTransitions",
    "StableSegment",
    "TrackedBand",
    "TrackedFrame",
    "TransitionBlock",
    "ValidationResult",
    "build_transition_blocks",
    "build_stable_segments",
    "characterize_band",
    "coupled_area_model",
    "empirical_parameters",
    "label_dilute_bands",
    "negative_log_likelihood",
    "optimize_parameters",
    "positive_parameters",
    "prepare_training_transitions",
    "raw_parameters",
    "run_analysis",
    "run_bayesian_inference",
    "stable_lags",
    "transition",
    "transition_log_density",
    "validate_posterior",
]
