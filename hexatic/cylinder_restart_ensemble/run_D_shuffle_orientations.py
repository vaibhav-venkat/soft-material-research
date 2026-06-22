from __future__ import annotations

if __package__:
    from .common import ORIGINAL_U0, RUN_STEPS, run_condition, shuffle_orientations, simulation
else:
    from common import ORIGINAL_U0, RUN_STEPS, run_condition, shuffle_orientations, simulation


SEED = simulation.seed


def main() -> None:
    run_condition(
        "D",
        transform=shuffle_orientations,
        active_u0=ORIGINAL_U0,
        seed=SEED,
        run_steps=RUN_STEPS,
    )


if __name__ == "__main__":
    main()
