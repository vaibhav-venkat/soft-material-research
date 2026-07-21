"""Minimal HOOMD-blue GPU smoke test.

Run with: pixi run python temp.py
"""

from __future__ import annotations

import itertools

import hoomd


def main() -> None:
    if not hoomd.device.GPU.is_available():
        reasons = hoomd.device.GPU.get_unavailable_device_reasons()
        detail = "\n".join(f"  - {reason}" for reason in reasons)
        raise SystemExit(f"HOOMD cannot use a GPU.\n{detail}")

    device = hoomd.device.GPU()
    print(f"HOOMD version: {hoomd.version.version}")
    print(f"GPU: {device.device}")

    simulation = hoomd.Simulation(device=device, seed=42)
    snapshot = hoomd.Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        positions = list(itertools.product((-3.0, -1.0, 1.0, 3.0), repeat=3))
        snapshot.configuration.box = [12.0, 12.0, 12.0, 0.0, 0.0, 0.0]
        snapshot.particles.N = len(positions)
        snapshot.particles.types = ["A"]
        snapshot.particles.position[:] = positions

    simulation.create_state_from_snapshot(snapshot)

    neighbor_list = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=neighbor_list)
    lj.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
    lj.r_cut[("A", "A")] = 2.5

    integrator = hoomd.md.Integrator(dt=0.001)
    integrator.methods.append(
        hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
    )
    integrator.forces.append(lj)
    simulation.operations.integrator = integrator

    simulation.run(30)
    print(f"Success: completed {simulation.timestep} steps on the GPU.")


if __name__ == "__main__":
    main()
