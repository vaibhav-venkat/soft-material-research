from __future__ import annotations

if __package__:
    from .common import RUN_STEPS, run_condition, simulation
else:
    from common import RUN_STEPS, run_condition, simulation


SEED = simulation.seed


def main() -> None:
    run_condition(
        "S",
        active_u0=0.0,
        seed=SEED,
        run_steps=RUN_STEPS,
    )


if __name__ == "__main__":
    main()
