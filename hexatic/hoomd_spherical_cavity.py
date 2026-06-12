import itertools
import math

import gsd.hoomd
import hoomd
import numpy as np

if __package__:
    from hexatic.constants import sphere
else:
    from constants import sphere

np.random.seed(sphere.SEED)

spacing = sphere.RHO ** (-1 / 3)
K = math.ceil(sphere.N_PARTICLES ** (1 / 3))
L = K * spacing
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = (
    np.asarray(list(itertools.product(x, repeat=3)))
    + np.array([spacing, spacing, spacing]) / 2
)

frame = gsd.hoomd.Frame()
frame.particles.N = sphere.N_PARTICLES
frame.particles.position = position
frame.particles.diameter = [sphere.PARTICLE_DIAMETER] * sphere.N_PARTICLES
# set orientation & MoI
frame.particles.moment_inertia = [(1, 1, 1)] * sphere.N_PARTICLES
orientation = 2.0 * np.pi * np.random.rand(sphere.N_PARTICLES)
phi = 2.0 * np.pi * np.random.rand(sphere.N_PARTICLES)
theta = 1.0 * np.pi * np.random.rand(sphere.N_PARTICLES)
orient_quat = [
    (
        np.cos(orientation[i] / 2),
        np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.cos(phi[i]),
        np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.sin(phi[i]),
        np.sin(orientation[i] / 2) * np.cos(theta[i]),
    )
    for i in range(sphere.N_PARTICLES)
]
frame.particles.orientation = orient_quat
frame.particles.typeid = [0] * sphere.N_PARTICLES
frame.configuration.box = [
    sphere.VOLUME ** (1 / 3) * 2,
    sphere.VOLUME ** (1 / 3) * 2,
    sphere.VOLUME ** (1 / 3) * 2,
    0,
    0,
    0,
]
frame.particles.types = ["A"]


sphere.INITIAL_GSD.parent.mkdir(parents=True, exist_ok=True)
with gsd.hoomd.open(name=str(sphere.INITIAL_GSD), mode="w") as f:
    f.append(frame)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=sphere.SEED)
state = sim.create_state_from_gsd(filename=str(sphere.INITIAL_GSD))

integrator = hoomd.md.Integrator(dt=sphere.TIMESTEP)
sim.operations.integrator = integrator
filter_all = hoomd.filter.All()

ov = hoomd.md.methods.OverdampedViscous(filter=filter_all)
integrator.methods.append(ov)
# brownian = hoomd.md.methods.Brownian(kT=kT, filter=filter_all)
# integrator.methods.append(brownian)

cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("A", "A")] = dict(
    epsilon=50 * sphere.GAMMA * sphere.U0 * sphere.SIGMA,
    sigma=sphere.SIGMA,
)
lj.r_cut[("A", "A")] = sphere.CUTOFF
integrator.forces.append(lj)

walls = [hoomd.wall.Sphere(radius=sphere.CAVITY_RADIUS)]
lj2 = hoomd.md.external.wall.LJ(walls=walls)
lj2.params["A"] = {
    "sigma": sphere.SIGMA,
    "epsilon": 50 * sphere.GAMMA * sphere.U0 * sphere.SIGMA,
    "r_cut": sphere.CUTOFF,
}


integrator.forces.append(lj2)


active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
active.use_orientation = True
active.active_force["A"] = (sphere.GAMMA * sphere.U0, 0.0, 0.0)  # will be rotated
active.active_torque["A"] = (0.0, 0.0, 0.0)
integrator.forces.append(active)


# Rotational diffusion updater
rot_diff = active.create_diffusion_updater(
    trigger=hoomd.trigger.Periodic(sphere.ROTATIONAL_DIFFUSION_PERIOD),
    rotational_diffusion=1 / sphere.TAU_R,
)
sim.operations += rot_diff

gsd_writer = hoomd.write.GSD(
    filename=str(sphere.IN_GSD),
    trigger=hoomd.trigger.Periodic(sphere.TRAJECTORY_WRITE_PERIOD),
    mode="wb",
)
sim.operations.writers.append(gsd_writer)

sim.run(sphere.RUN_STEPS)
