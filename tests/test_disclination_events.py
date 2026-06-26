from __future__ import annotations

import numpy as np

from hexatic.multiple_sim_analysis.disclination_order_fields.events.constants import (
    EventAnalysisConstants,
)
from hexatic.multiple_sim_analysis.disclination_order_fields.events.events import (
    classify_defect_events,
)
from hexatic.multiple_sim_analysis.disclination_order_fields.events.fields import (
    compute_d2min_frame,
    sample_defect_annulus_fields,
)
from hexatic.multiple_sim_analysis.disclination_order_fields.events.runner import (
    cluster_defects_frame,
)
from hexatic.multiple_sim_analysis.disclination_order_fields.events.tracking import (
    DefectFrame,
    DefectTrackTable,
    _match_defects_frame,
)


def _frame(frame_index, x_values, charges):
    positions = np.column_stack(
        (
            np.asarray(x_values, dtype=np.float64),
            np.zeros(len(x_values), dtype=np.float64),
            np.ones(len(x_values), dtype=np.float64),
        )
    )
    return DefectFrame(
        frame_index=frame_index,
        step=frame_index,
        particle_indices=np.arange(len(x_values), dtype=np.int64),
        charges=np.asarray(charges, dtype=np.int64),
        positions=positions,
        cartesian_positions=np.column_stack(
            (
                np.asarray(x_values, dtype=np.float64),
                np.zeros(len(x_values), dtype=np.float64),
                np.ones(len(x_values), dtype=np.float64),
            )
        ),
    )


def test_match_defects_frame_confident_same_charge_pairs():
    prev = _frame(0, [0.0, 5.0], [1, -1])
    cur = _frame(1, [0.3, 5.2], [1, -1])

    match = _match_defects_frame(prev, cur, 20.0, match_tol=1.0)

    pairs = sorted(zip(match.prev_indices, match.cur_indices, match.distances))
    assert [(prev_idx, cur_idx) for prev_idx, cur_idx, _ in pairs] == [(0, 0), (1, 1)]
    np.testing.assert_allclose([distance for _, _, distance in pairs], [0.3, 0.2])
    assert match.ambiguous_pair_count == 0


def test_match_defects_frame_rejects_ambiguous_swaps():
    prev = _frame(0, [0.0, 1.0], [1, 1])
    cur = _frame(1, [0.45, 0.55], [1, 1])

    match = _match_defects_frame(prev, cur, 20.0, match_tol=1.0)

    assert match.prev_indices.size == 0
    assert match.cur_indices.size == 0
    assert match.rejected_swap_count > 0


def test_event_classification_birth_death_and_annihilation_pair():
    frame_axis = np.asarray([0, 1, 2, 3], dtype=np.int64)
    positions = np.full((2, 4, 3), np.nan, dtype=np.float64)
    particle_index = np.full((2, 4), -1, dtype=np.int64)

    particle_index[0, 1:3] = 0
    positions[0, 1] = [0.5, 0.0, 1.0]
    positions[0, 2] = [0.6, 0.0, 1.0]

    particle_index[1, 0:3] = 1
    positions[1, 0] = [0.0, 0.0, 1.0]
    positions[1, 1] = [0.4, 0.0, 1.0]
    positions[1, 2] = [0.7, 0.0, 1.0]

    table = DefectTrackTable(
        frame_axis=frame_axis,
        steps=frame_axis.copy(),
        track_id=np.asarray([0, 1], dtype=np.int64),
        charge=np.asarray([1, -1], dtype=np.int64),
        frame_start=np.asarray([1, 0], dtype=np.int64),
        frame_stop=np.asarray([3, 3], dtype=np.int64),
        particle_index=particle_index,
        positions=positions,
        matched_confident=particle_index >= 0,
        velocity=np.full((2, 4, 3), np.nan, dtype=np.float64),
        birth_ambiguous=np.asarray([False, False]),
        death_ambiguous=np.asarray([False, False]),
        annihilation_flag=np.asarray([False, False]),
        annihilation_partner_id=np.asarray([-1, -1], dtype=np.int64),
        ambiguity_count_by_frame=np.zeros(4, dtype=np.int64),
        rejected_swap_count_by_frame=np.zeros(4, dtype=np.int64),
    )

    events = classify_defect_events(
        table,
        20.0,
        constants=EventAnalysisConstants(neighbor_count_radius=1.0),
    )

    np.testing.assert_array_equal(events.birth_track_id, [0])
    np.testing.assert_array_equal(events.death_track_id, [0, 1])
    np.testing.assert_array_equal(events.death_annihilated, [True, True])
    np.testing.assert_array_equal(events.death_partner_track_id, [1, 0])


def test_sample_defect_annulus_fields_returns_nan_for_empty_annulus():
    coords = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [5.0, 0.0, 1.0],
        ]
    )
    charges = np.asarray([1, 0, 0])

    sample = sample_defect_annulus_fields(
        coords,
        coords[0],
        charges,
        20.0,
        scalar_fields={"rho": np.asarray([1.0, 2.0, 3.0])},
        constants=EventAnalysisConstants(
            annulus_core_radius=1.0,
            annulus_outer_radius=2.0,
        ),
    )

    assert sample["annulus_count"] == 0.0
    assert np.isnan(sample["rho"])


def test_d2min_affine_deformation_is_zero_for_center_particle():
    radius = 10.0
    physical = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    shear = np.asarray([[1.0, 0.25], [0.0, 1.0]])
    deformed = physical @ shear.T
    coords_t = np.column_stack((physical[:, 0], physical[:, 1] / radius, np.full(5, radius)))
    coords_next = np.column_stack(
        (deformed[:, 0], deformed[:, 1] / radius, np.full(5, radius))
    )

    d2min = compute_d2min_frame(
        coords_t,
        coords_next,
        radius=radius,
        box_length_x=100.0,
        neighbor_radius=2.0,
        normalization_length=1.0,
        min_neighbors=3,
    )

    assert d2min[0] < 1e-12


def test_cluster_defects_frame_uses_periodic_x_bonds():
    constants = EventAnalysisConstants(cluster_bond_length=2.1)
    positions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [8.1, 0.0, 1.0],
        ]
    )
    charges = np.asarray([1, -1, 1])

    clusters = cluster_defects_frame(positions, charges, 10.0, constants=constants)

    assert clusters.cluster_id.size == 1
    np.testing.assert_array_equal(clusters.size, [3])
    np.testing.assert_array_equal(clusters.charge, [1])
