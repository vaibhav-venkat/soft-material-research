from __future__ import annotations

import numpy as np


def validate_frame_particle_shape(
    name: str,
    values: np.ndarray,
    expected_shape: tuple[int, int],
) -> None:
    if np.asarray(values).shape[:2] != expected_shape:
        raise ValueError(
            f"{name} shape {np.asarray(values).shape[:2]} does not match "
            f"expected frame/particle shape {expected_shape}."
        )


def validate_step_alignment(
    name: str,
    steps: np.ndarray,
    expected_steps: np.ndarray,
) -> None:
    steps = np.asarray(steps, dtype=np.int64)
    expected_steps = np.asarray(expected_steps, dtype=np.int64)
    if steps.shape != expected_steps.shape or np.any(steps != expected_steps):
        raise ValueError(f"{name} steps do not align with expected frame steps.")
