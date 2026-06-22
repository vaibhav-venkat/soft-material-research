from __future__ import annotations

if __package__:
    from .common import (
        ORIGINAL_U0,
        RUN_STEPS,
        condition_paths,
        ensure_paths_available,
        random_hoomd_seed,
        run_condition,
    )
else:
    from common import (
        ORIGINAL_U0,
        RUN_STEPS,
        condition_paths,
        ensure_paths_available,
        random_hoomd_seed,
        run_condition,
    )


N_REPLICAS = 1


def main() -> None:
    targets = []
    seeds = [random_hoomd_seed() for _ in range(N_REPLICAS)]
    for replica in range(N_REPLICAS):
        targets.extend(condition_paths("A", replica=replica))
    ensure_paths_available(*targets)

    for replica, seed in enumerate(seeds):
        run_condition(
            "A",
            active_u0=ORIGINAL_U0,
            seed=seed,
            replica=replica,
            run_steps=RUN_STEPS,
        )


if __name__ == "__main__":
    main()
