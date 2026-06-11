import hoomd
import gsd.hoomd
import numpy as np
import itertools
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CYLINDER_OUTPUT_DIR = PROJECT_DIR / "output" / "cylinder"

N = 4000
rho = 0.2
V = N / rho
xrat = 4
L = (V / xrat)**(1/3)
Lx = xrat * L
K = math.ceil((N / xrat)** (1 / 3))
Kx = xrat * K
l = rho**(-1/3)/xrat
lx = xrat * l

N = Kx * K**2

seed = 1
kT = 1
sigma = 1
gamma = 1
U0 = 100
tauR = 1

np.random.seed(seed)

# spacing = rho**(-1/3)
# K = math.ceil(N ** (1 / 3))
# L = K * spacing
# x = np.linspace(-L / 2, L / 2, K, endpoint=False)
# position = np.asarray(list(itertools.product(x, repeat=3))) + np.array([spacing, spacing, spacing]) / 2


def generate_lattice(nx, ny, nz, lx, ly, lz):
    # 1. Create 1D arrays for each axis
    x = np.arange(nx) * lx
    y = np.arange(ny) * ly
    z = np.arange(nz) * lz
    
    # 2. Generate the 3D grid
    # Use indexing='ij' to ensure the output matches (x, y, z) order
    xv, yv, zv = np.meshgrid(x - np.mean(x), y - np.mean(y), z - np.mean(z), indexing='ij')
    
    # 3. Stack into an (N, 3) array of coordinates
    coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
    return coords

# Example: 4x4x4 grid with spacings 1.5, 2.0, and 1.0
position = generate_lattice(Kx, K, K, 2**(1/6)+0.5, 2**(1/6)+0.5, 2**(1/6)+0.5)

print(N)

print(np.max(position, axis=0))
print(np.min(position, axis=0))
Lx = np.max(position, axis=0)[0]*2 + 2
L = 0.4 * Lx
print(Lx/2, L/2, L/2)
print(10 * 2**(1/6))
print(np.max(np.sqrt(position[:, 2]**2 + position[:, 1]**2)))
# assert False


frame = gsd.hoomd.Frame()
frame.particles.N = N
frame.particles.position = position
frame.particles.diameter = [sigma * 2.**(1. / 6.)] * N
# set orientation & MoI
frame.particles.moment_inertia = [(1, 1, 1)] * N
orientation = 2.0 * np.pi * np.random.rand(N)
phi = 2.0 * np.pi * np.random.rand(N)
theta = 1.0 * np.pi * np.random.rand(N)
orient_quat = [(np.cos(orientation[i] / 2),
                np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.cos(phi[i]),
                np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.sin(phi[i]),
                np.sin(orientation[i] / 2) * np.cos(theta[i])) for i in range(N)]
frame.particles.orientation = orient_quat
frame.particles.typeid = [0] * N
frame.configuration.box = [Lx, L, L, 0, 0, 0]
frame.particles.types = ["A"]


CYLINDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
initial_gsd = CYLINDER_OUTPUT_DIR / "initial_mesh.gsd"
with gsd.hoomd.open(name=str(initial_gsd), mode="w") as f:
    f.append(frame)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=seed)
state = sim.create_state_from_gsd(filename=str(initial_gsd))

integrator = hoomd.md.Integrator(dt=1e-6)
sim.operations.integrator = integrator
filter_all = hoomd.filter.All()

ov = hoomd.md.methods.OverdampedViscous(filter=filter_all)
integrator.methods.append(ov)
# brownian = hoomd.md.methods.Brownian(kT=kT, filter=filter_all)
# integrator.methods.append(brownian)

cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[("A", "A")] = dict(epsilon=50 * gamma * U0 * sigma, sigma=sigma)
lj.r_cut[("A", "A")] = 2 ** (1.0 / 6.0) * sigma
integrator.forces.append(lj)

# walls = [hoomd.wall.Sphere(radius=1.4 * (V * 3 / 4 / np.pi)**(1/3))]
walls = [hoomd.wall.Cylinder(radius=10 * 2**(1/6), axis=(1, 0, 0), inside=True, open=True)]
lj2 = hoomd.md.external.wall.LJ(walls=walls)
lj2.params["A"] = {
    "sigma": sigma,
    "epsilon": 50 * gamma * U0 * sigma,
    "r_cut": 2 ** (1.0 / 6.0) * sigma,
}


integrator.forces.append(lj2)


active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
active.use_orientation = True
active.active_force['A'] = (gamma * U0, 0.0, 0.0)  # will be rotated
active.active_torque['A'] = (0.0, 0.0, 0.0)
integrator.forces.append(active)


# Rotational diffusion updater
rot_diff = active.create_diffusion_updater(
    trigger=hoomd.trigger.Periodic(10),
    rotational_diffusion=1/tauR
)
sim.operations += rot_diff

gsd_writer = hoomd.write.GSD(
    filename=str(CYLINDER_OUTPUT_DIR / "trajectory_cylinder.gsd"),
    trigger=hoomd.trigger.Periodic(int(1e5)),
    mode="wb",
)
# gsd_writer = hoomd.write.GSD(
#     filename="trajectory.gsd", trigger=hoomd.trigger.Periodic(1), mode="wb"
# )
sim.operations.writers.append(gsd_writer)

sim.run(int(1e7))
# sim.run(10)
