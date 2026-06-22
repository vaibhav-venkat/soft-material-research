from __future__ import annotations

if __package__:
    from .common import ORIGINAL_U0, RUN_STEPS, run_condition, simulation, theta_mirror
else:
    from common import ORIGINAL_U0, RUN_STEPS, run_condition, simulation, theta_mirror


SEED = simulation.seed


def main() -> None:
    run_condition(
        "J",
        transform=theta_mirror,
        active_u0=ORIGINAL_U0,
        seed=SEED,
        run_steps=RUN_STEPS,
    )


if __name__ == "__main__":
    main()
