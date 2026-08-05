"""Stable dilute-band area extraction and inference."""

from .detection.components import BandComponent, label_dilute_bands
from .fitting.bayesian import (
    BayesianResult,
    MCMCConfig,
    MCMCDiagnostics,
    coupled_area_model,
    run_bayesian_inference,
)
from .detection.characterization import DetectedBand, characterize_band
from .detection.density import SurfaceGrid
from .fitting.inference import (
    HessianDiagnostics,
    OptimizationResult,
    OptimizationRun,
    empirical_parameters,
    optimize_parameters,
)
from .fitting.model import (
    PARAMETER_NAMES,
    JITTER,
    PersistentRateBlock,
    Scaling,
    StateSpaceSequence,
    TrainingTransitions,
    TransitionBlock,
    build_transition_blocks,
    build_state_space_sequences,
    build_persistent_rate_blocks,
    conservative_segment_slope,
    conservative_segment_slopes,
    estimate_slope_variance,
    negative_log_likelihood,
    parameter_negative_log_likelihood,
    persistent_innovations,
    positive_parameters,
    prepare_training_transitions,
    raw_parameters,
    transition,
    transition_log_density,
)
from .fitting.masked_model import (
    MaskedPersistentRateBlock,
    MaskedSlope,
    MaskedTrainingTransitions,
    MaskedTransitionBlock,
    allocation as masked_allocation,
    build_persistent_rate_blocks as build_masked_persistent_rate_blocks,
    build_transition_blocks as build_masked_transition_blocks,
    conservative_projection as masked_conservative_projection,
    conservative_run_slopes,
    estimate_slope_variance as estimate_masked_slope_variance,
    negative_log_likelihood as masked_negative_log_likelihood,
    normalize_event_chain,
    parameter_negative_log_likelihood as masked_parameter_negative_log_likelihood,
    prepare_training_transitions as prepare_masked_training_transitions,
    propagate_event,
    scaling_from_chains,
    transition as masked_transition,
    transition_log_density as masked_transition_log_density,
)
from .detection.chains import (
    ChainPlan,
    EventChain,
    KnotGrid,
    build_chains,
    build_event_chains,
    build_knot_grid,
    build_knot_grids,
    event_transitions,
    global_n_max,
    plan_chains,
)
from .detection.events import (
    EventCompositionError,
    EventStep,
    build_event_step,
    initial_slots,
    no_event_step,
    pack_areas,
    slot_mask,
)
from .detection.segments import StableSegment, build_stable_segments
from .detection.tracking import (
    BandTracker,
    DetectionFrame,
    EventCode,
    EventEdges,
    EventRecord,
    TrackedBand,
    TrackedFrame,
)
from .fitting.validation import ValidationResult, validate_posterior
from .fitting.event_validation import (
    EventSimulation,
    simulate_event_chain,
    validate_event_posterior,
)
from .pipeline.workflow import AnalysisConfig, LagOutcome, run_analysis, stable_lags
