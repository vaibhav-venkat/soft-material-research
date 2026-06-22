from __future__ import annotations

if __package__:
    from .common import (
        ORIGINAL_U0,
        RUN_STEPS,
        consistent_axial_mirror,
        run_condition,
        simulation,
    )
else:
    from common import (
        ORIGINAL_U0,
        RUN_STEPS,
        consistent_axial_mirror,
        run_condition,
        simulation,
    )


SEED = simulation.seed


def main() -> None:
    run_condition(
        "H",
        transform=consistent_axial_mirror,
        active_u0=ORIGINAL_U0,
        seed=SEED,
        run_steps=RUN_STEPS,
    )


if __name__ == "__main__":
    main()
