import math

import gsd.hoomd
import hoomd
import numpy as np

if __package__:
    from hexatic.constants import cylinder
else:
    from constants import cylinder

paths = cylinder.PATHS
analysis = cylinder.ANALYSIS
simulation = cylinder.SIMULATION

RUN_STEPS = int(1e7)
TRAJECTORY_WRITE_PERIOD = int(1e5)
OUTPUT_GSD = paths.in_gsd.parent / "trajectory_cylinder_new_rad.gsd"
INITIAL_GSD = paths.in_gsd.parent / "initial_mesh_cylinder_new_rad.gsd"
NEW_TIMESTEP = simulation.timestep / 10.0

NEW_CIRCUMFERENCE = 60.0 * analysis.particle_diameter
NEW_RADIUS = NEW_CIRCUMFERENCE / (2.0 * math.pi)
TARGET_N = int(simulation.requested_n_particles)
TARGET_VOLUME = TARGET_N / float(simulation.rho)
Lx = TARGET_VOLUME / (math.pi * NEW_RADIUS**2)
TRANSVERSE_MARGIN = 8.0 * analysis.wall_cutoff + 2.0 * analysis.particle_diameter
L = 2.0 * (NEW_RADIUS + TRANSVERSE_MARGIN)


def cylinder_cross_section_points(radius, min_spacing):
    center_limit = max(
        0.0,
        radius - analysis.wall_cutoff - 0.001 * analysis.particle_diameter,
    )
    if center_limit == 0.0:
        return np.asarray([[0.0, 0.0]], dtype=np.float64)

    n_radial = max(1, int(math.ceil(center_limit / min_spacing)) + 1)
    candidates = []
    for ring_radius in np.linspace(center_limit, 0.0, n_radial):
        if ring_radius == 0.0:
            candidates.append((0.0, 0.0))
            continue
        n_angles = max(12, int(math.ceil(2.0 * math.pi * ring_radius / min_spacing)) * 4)
        for angle in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
            candidates.append(
                (
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                )
            )

    selected = []
    min_distance_squared = min_spacing**2
    for candidate in candidates:
        if all(
            (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2
            >= min_distance_squared
            for point in selected
        ):
            selected.append(candidate)

    if not selected:
        selected.append((0.0, 0.0))
    return np.asarray(selected, dtype=np.float64)


def generate_cylinder_lattice(n_particles, box_length_x, radius, spacing):
    cross_section = cylinder_cross_section_points(radius, spacing)
    n_x = int(math.ceil(n_particles / len(cross_section)))
    dx = box_length_x / n_x
    x_values = -0.5 * box_length_x + (np.arange(n_x, dtype=np.float64) + 0.5) * dx
    positions = np.empty((n_x * len(cross_section), 3), dtype=np.float64)

    particle_idx = 0
    for x in x_values:
        for y, z in cross_section:
            positions[particle_idx] = (x, y, z)
            particle_idx += 1

    return positions[:n_particles]


def random_uniform_quaternions(n, rng=np.random):
    u1 = rng.rand(n)
    u2 = rng.rand(n)
    u3 = rng.rand(n)

    qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    q = np.column_stack([qw, qx, qy, qz])
    q /= np.linalg.norm(q, axis=1)[:, None]
    return q


np.random.seed(simulation.seed)

position = generate_cylinder_lattice(
    TARGET_N,
    Lx,
    NEW_RADIUS,
    analysis.wall_cutoff,
)
N = position.shape[0]

frame = gsd.hoomd.Frame()
frame.particles.N = N
frame.particles.position = position
frame.particles.diameter = [analysis.particle_diameter] * N
frame.particles.moment_inertia = [(1, 1, 1)] * N
frame.particles.orientation = random_uniform_quaternions(N)
frame.particles.image = np.zeros((N, 3), dtype=np.int32)
frame.particles.typeid = [0] * N
frame.configuration.box = [Lx, L, L, 0, 0, 0]
frame.configuration.dimensions = 3
frame.particles.types = ["A"]

INITIAL_GSD.parent.mkdir(parents=True, exist_ok=True)
with gsd.hoomd.open(name=str(INITIAL_GSD), mode="w") as f:
    f.append(frame)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=simulation.seed)
sim.create_state_from_gsd(filename=str(INITIAL_GSD))

integrator = hoomd.md.Integrator(dt=NEW_TIMESTEP)
sim.operations.integrator = integrator
filter_all = hoomd.filter.All()

ov = hoomd.md.methods.OverdampedViscous(filter=filter_all)
integrator.methods.append(ov)

cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("A", "A")] = dict(
    epsilon=50 * simulation.gamma * simulation.u0 * analysis.sigma,
    sigma=analysis.sigma,
)
lj.r_cut[("A", "A")] = analysis.wall_cutoff
integrator.forces.append(lj)

walls = [
    hoomd.wall.Cylinder(
        radius=NEW_RADIUS,
        axis=(1, 0, 0),
        inside=True,
        open=True,
    )
]
lj2 = hoomd.md.external.wall.LJ(walls=walls)
lj2.params["A"] = {
    "sigma": analysis.sigma,
    "epsilon": 50 * simulation.gamma * simulation.u0 * analysis.sigma,
    "r_cut": analysis.wall_cutoff,
}
integrator.forces.append(lj2)

active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
active.use_orientation = True
active.active_force["A"] = (simulation.gamma * simulation.u0, 0.0, 0.0)
active.active_torque["A"] = (0.0, 0.0, 0.0)
integrator.forces.append(active)

rot_diff = active.create_diffusion_updater(
    trigger=hoomd.trigger.Periodic(simulation.rotational_diffusion_period),
    rotational_diffusion=1 / simulation.tau_r,
)
sim.operations += rot_diff

logger = hoomd.logging.Logger(categories=["particle"])
logger.add(lj, quantities=["forces", "virials"])
logger.add(lj2, quantities=["forces", "virials"])

gsd_writer = hoomd.write.GSD(
    filename=str(OUTPUT_GSD),
    trigger=hoomd.trigger.Periodic(TRAJECTORY_WRITE_PERIOD),
    mode="wb",
    dynamic=["property", "particles/orientation"],
    logger=logger,
)
sim.operations.writers.append(gsd_writer)
print(f"running for {RUN_STEPS}")
sim.run(RUN_STEPS)
