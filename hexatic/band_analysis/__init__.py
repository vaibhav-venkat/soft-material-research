"""Stable dilute-band area extraction and inference."""

from .components import BandComponent, label_dilute_bands
from .characterization import DetectedBand, characterize_band
from .density import SurfaceGrid
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
    "SurfaceGrid",
    "StableSegment",
    "TrackedBand",
    "TrackedFrame",
    "build_stable_segments",
    "characterize_band",
    "label_dilute_bands",
]
