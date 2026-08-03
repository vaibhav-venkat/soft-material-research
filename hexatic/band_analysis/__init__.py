"""Stable dilute-band area extraction and inference."""

from .components import BandComponent, label_dilute_bands
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

__all__ = [
    "BandComponent",
    "BandTracker",
    "DetectedBand",
    "DetectionFrame",
    "EventCode",
    "EventRecord",
    "HessianDiagnostics",
    "JITTER",
    "OptimizationResult",
    "OptimizationRun",
    "PARAMETER_NAMES",
    "Scaling",
    "SurfaceGrid",
    "TrainingTransitions",
    "StableSegment",
    "TrackedBand",
    "TrackedFrame",
    "TransitionBlock",
    "build_transition_blocks",
    "build_stable_segments",
    "characterize_band",
    "empirical_parameters",
    "label_dilute_bands",
    "negative_log_likelihood",
    "optimize_parameters",
    "positive_parameters",
    "prepare_training_transitions",
    "raw_parameters",
    "transition",
    "transition_log_density",
]
