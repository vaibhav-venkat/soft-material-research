import itertools
import math

import gsd.hoomd
import hoomd
import numpy as np

if __package__:
    from hexatic.constants import sphere
else:
    from constants import sphere

paths = sphere.PATHS
analysis = sphere.ANALYSIS
simulation = sphere.SIMULATION

np.random.seed(simulation.seed)

spacing = simulation.rho ** (-1 / 3)
K = math.ceil(simulation.n_particles ** (1 / 3))
L = K * spacing
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = (
    np.asarray(list(itertools.product(x, repeat=3)))
    + np.array([spacing, spacing, spacing]) / 2
)

frame = gsd.hoomd.Frame()
frame.particles.N = simulation.n_particles
frame.particles.position = position
frame.particles.diameter = [analysis.particle_diameter] * simulation.n_particles
# set orientation & MoI
frame.particles.moment_inertia = [(1, 1, 1)] * simulation.n_particles
orientation = 2.0 * np.pi * np.random.rand(simulation.n_particles)
phi = 2.0 * np.pi * np.random.rand(simulation.n_particles)
theta = 1.0 * np.pi * np.random.rand(simulation.n_particles)
orient_quat = [
    (
        np.cos(orientation[i] / 2),
        np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.cos(phi[i]),
        np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.sin(phi[i]),
        np.sin(orientation[i] / 2) * np.cos(theta[i]),
    )
    for i in range(simulation.n_particles)
]
frame.particles.orientation = orient_quat
frame.particles.typeid = [0] * simulation.n_particles
frame.configuration.box = [
    analysis.volume ** (1 / 3) * 2,
    analysis.volume ** (1 / 3) * 2,
    analysis.volume ** (1 / 3) * 2,
    0,
    0,
    0,
]
frame.particles.types = ["A"]


paths.initial_gsd.parent.mkdir(parents=True, exist_ok=True)
with gsd.hoomd.open(name=str(paths.initial_gsd), mode="w") as f:
    f.append(frame)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=simulation.seed)
state = sim.create_state_from_gsd(filename=str(paths.initial_gsd))

integrator = hoomd.md.Integrator(dt=simulation.timestep)
sim.operations.integrator = integrator
filter_all = hoomd.filter.All()

ov = hoomd.md.methods.OverdampedViscous(filter=filter_all)
integrator.methods.append(ov)
# brownian = hoomd.md.methods.Brownian(kT=kT, filter=filter_all)
# integrator.methods.append(brownian)

cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("A", "A")] = dict(
    epsilon=50 * simulation.gamma * simulation.u0 * analysis.sigma,
    sigma=analysis.sigma,
)
lj.r_cut[("A", "A")] = analysis.cutoff
integrator.forces.append(lj)

walls = [hoomd.wall.Sphere(radius=analysis.cavity_radius)]
lj2 = hoomd.md.external.wall.LJ(walls=walls)
lj2.params["A"] = {
    "sigma": analysis.sigma,
    "epsilon": 50 * simulation.gamma * simulation.u0 * analysis.sigma,
    "r_cut": analysis.cutoff,
}


integrator.forces.append(lj2)


active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
active.use_orientation = True
active.active_force["A"] = (simulation.gamma * simulation.u0, 0.0, 0.0)  # will be rotated
active.active_torque["A"] = (0.0, 0.0, 0.0)
integrator.forces.append(active)


# Rotational diffusion updater
rot_diff = active.create_diffusion_updater(
    trigger=hoomd.trigger.Periodic(simulation.rotational_diffusion_period),
    rotational_diffusion=1 / simulation.tau_r,
)
sim.operations += rot_diff

gsd_writer = hoomd.write.GSD(
    filename=str(paths.in_gsd),
    trigger=hoomd.trigger.Periodic(simulation.trajectory_write_period),
    mode="wb",
)
sim.operations.writers.append(gsd_writer)

sim.run(simulation.run_steps)
