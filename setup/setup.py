# ideal 2D vesicle (from modified parameter version) + perimeter conservation

import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np
from hoomd import update
from hoomd.custom import Action
from hoomd.md.external import wall
from hoomd.md.force import Custom

# Parameters
N_vertex = 250
N_abp = 100

sigma_vertex = 0.05
sigma_abp = 1.0
min_dist = 0.5 * (sigma_vertex + sigma_abp)
l0 = 4 * sigma_vertex
delta = 0.5 * l0  # buffer for cutoff range

R_vertex = 0.5 * N_vertex * l0 / np.pi
D0 = R_vertex * 2


R_abp = 0.3
# R_vertex, R_abp not particle radius but radial distance

kT = 1.0
dt = 0.000001
gamma = 1

# k_bond = 15000
k_bond = 1
k_bend = 20
v0 = 20
tau = 1

print(v0 * tau / D0)
# assert False

P0 = N_vertex * l0
k_p = 100

seed = 42
np.random.seed(seed)

# ---- Create vertex ring ----
theta = np.linspace(0, 2 * np.pi, N_vertex, endpoint=False)
vertex_pos = np.array(
    [(R_vertex * np.cos(t), R_vertex * np.sin(t), 0.0) for t in theta]
)
P0 = 2 * np.pi * R_vertex

# ---- Place ABPs inside ring ----
abp_pos = []
while len(abp_pos) < N_abp:
    r = R_vertex * np.sqrt(np.random.rand()) * 0.9
    angle = 2 * np.pi * np.random.rand()
    x, y = r * np.cos(angle), r * np.sin(angle)
    if np.min(np.linalg.norm(vertex_pos[:, :2] - [x, y], axis=1)) > min_dist:
        abp_pos.append((x, y, 0.0))

# Combine
positions = np.vstack((vertex_pos, abp_pos))
N_particles = len(positions)
types = ["vertex", "abp"]
typeid = [0] * N_vertex + [1] * N_abp
diameters = [sigma_vertex if tid == 0 else sigma_abp for tid in typeid]

# Random 2D orientation for all particles
orient_angle = 2 * np.pi * np.random.rand(N_particles)
quats = [(np.cos(a / 2), 0, 0, np.sin(a / 2)) for a in orient_angle]

# --- Build GSD frame ---
frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = positions
frame.particles.orientation = quats
frame.particles.types = types
frame.particles.typeid = typeid
frame.particles.diameter = diameters
frame.particles.moment_inertia = [(0, 0, 1)] * N_particles
frame.configuration.box = [D0 * 2, D0 * 2, 0, 0, 0, 0]  # Lz=0 for 2D

# Bonds (close ring)
bonds = [(i, (i + 1) % N_vertex) for i in range(N_vertex)]
frame.bonds.N = len(bonds)
frame.bonds.types = ["bonds"]
frame.bonds.typeid = [0] * len(bonds)
frame.bonds.group = bonds

# for i, j in bonds:
#     dr = positions[i, :] - positions[j, :]
#     print(np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2) / sigma_vertex)

# assert False

# Angles (for curvature)
angles = [((i - 1) % N_vertex, i, (i + 1) % N_vertex) for i in range(N_vertex)]
frame.angles.N = len(angles)
frame.angles.types = ["harmonic"]
frame.angles.typeid = [0] * len(angles)
frame.angles.group = angles

with gsd.hoomd.open(name="initial-abp-vesicle.gsd", mode="w") as f:
    f.append(frame)

# ---- Simulation ----
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=seed)
sim.create_state_from_gsd("initial-abp-vesicle.gsd")

integrator = hoomd.md.Integrator(dt=dt)
sim.operations.integrator = integrator
all_particles = hoomd.filter.All()

# Langevin / Brownian dynamics
bd = hoomd.md.methods.Brownian(filter=all_particles, kT=kT)
bd.gamma.default = gamma
integrator.methods.append(bd)

# Tether Potential - modify from 3D version
bond_force = hoomd.md.bond.Tether()
bond_force.filter = hoomd.filter.Type(["vertex"])


bond_force.params["bonds"] = dict(
    k_b=k_bond,
    l_min=0,
    # l_min = sigma_vertex,
    l_c1=3.9999 * sigma_vertex,
    l_c0=4.0001 * sigma_vertex,
    l_max=10 * sigma_vertex,
)


integrator.forces.append(bond_force)


# perimeter_conservation_force = hoomd.md.bond.PerimeterConservation(k_p, P0)
# integrator.forces.append(perimeter_conservation_force)

# Bending angle (U = k*(1 - cos(θ)))
angle_force = hoomd.md.angle.Harmonic()
angle_force.params["harmonic"] = dict(k=k_bend, t0=np.pi)
integrator.forces.append(angle_force)

# LJ repulsion
nl = hoomd.md.nlist.Cell(buffer=0.4, exclusions=["bond", "angle"])
lj = hoomd.md.pair.LJ(nlist=nl)
lj.params[("vertex", "vertex")] = dict(
    epsilon=50 * gamma * v0 * sigma_vertex, sigma=sigma_vertex
)
lj.params[("vertex", "abp")] = dict(
    epsilon=50 * gamma * v0 * (sigma_vertex + sigma_abp) / 2,
    sigma=(sigma_vertex + sigma_abp) / 2,
)
lj.params[("abp", "abp")] = dict(epsilon=0 * gamma * v0 * sigma_abp, sigma=sigma_abp)

lj.r_cut[("vertex", "vertex")] = 2 ** (1 / 6) * sigma_vertex
lj.r_cut[("vertex", "abp")] = 2 ** (1 / 6) * (sigma_vertex + sigma_abp) / 2
lj.r_cut[("abp", "abp")] = 0.0

integrator.forces.append(lj)

# Active force for ABPs
active = hoomd.md.force.Active(filter=hoomd.filter.Type(["abp"]))
active.use_orientation = True
active.active_force["abp"] = (v0 * gamma, 0.0, 0.0)  # will be rotated
active.active_torque["abp"] = (0.0, 0.0, 0.0)
integrator.forces.append(active)

# Rotational diffusion updater
rot_diff = active.create_diffusion_updater(
    trigger=hoomd.trigger.Periodic(1), rotational_diffusion=1.0 / tau
)
sim.operations += rot_diff


# Perimeter conservation (2D perimeter; 3D area conservation)
# U_perimeter = (k_p * (L-L0) ** 2) / (2 * L0) = force acts to restore L to L0
class PerimeterConservation(Custom):
    def __init__(self, k_p, P0, vertex_typeid=0):
        super().__init__()
        self.k_p = k_p
        self.P0 = P0
        self.vertex_typeid = vertex_typeid

    def compute(self, timestep, positions, box, velocities=None, types=None):
        N_total = len(positions)
        forces = np.zeros((N_total, 3))
        # initializes zero force vector for all particles (only vertices)

        vertex_indices = np.where(types == self.vertex_typeid)[0]
        vertex_positions = positions[vertex_indices]

        def wrap(dr):
            dr[0] -= box.Lx * np.round(dr[0] / box.Lx)
            dr[1] -= box.Ly * np.round(dr[1] / box.Ly)
            return dr
            # periodic wrapping for shortest dr

        # only on vertex particles
        P = 0.0
        diffs = []
        for i in range(len(vertex_indices)):
            j = (i + 1) % len(vertex_indices)  # particle next to i
            dr = wrap(vertex_positions[j] - vertex_positions[i])
            diffs.append((vertex_indices[i], vertex_indices[j], dr))
            P += np.linalg.norm(dr)  # append to total perimeter

        dU_dP = self.k_p * (P - self.P0)  # dU = kp * (P - P0)

        for i, j, dr in diffs:
            norm_dr = np.linalg.norm(dr)
            if norm_dr > 0:
                f = dU_dP * dr / norm_dr
                forces[i] += f
                forces[j] -= f
                # equal and opposite forces to i, j for conserving momentum?

        return forces


# P0 = N_vertex * l0
# k_p = 100000
# perimeter_conservation = PerimeterConservation(k_p, P0, vertex_typeid=0)
# integrator.forces.append(perimeter_conservation)

# --- Output ---
filename = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-active.gsd"
# filename = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-passive.gsd"
gsd_writer = hoomd.write.GSD(
    filename=filename,
    trigger=hoomd.trigger.Periodic(5000),
    filter=hoomd.filter.All(),
    mode="wb",
)
sim.operations.writers.append(gsd_writer)

print("Running simulation...")
sim.run(1000000)
print("Done.")
