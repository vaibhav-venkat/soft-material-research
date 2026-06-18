import hoomd
import gsd.hoomd
import numpy as np

if __package__:
    from hexatic.constants import cylinder
else:
    from constants import cylinder

paths = cylinder.PATHS
analysis = cylinder.ANALYSIS
simulation = cylinder.SIMULATION


def copy_particle_property(target, source, name):
    value = getattr(source, name, None)
    if value is not None:
        if name == "types":
            setattr(target, name, list(value))
        else:
            setattr(target, name, np.asarray(value).copy())


def flipped_restart_frame(input_gsd):
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) > 0
        source_frame = source[-1]
        source_particles = source_frame.particles
        assert source_particles.position is not None
        assert source_particles.orientation is not None

        frame = gsd.hoomd.Frame()
        frame.configuration.step = int(source_frame.configuration.step)
        frame.configuration.box = np.asarray(source_frame.configuration.box, dtype=np.float64)
        dimensions = getattr(source_frame.configuration, "dimensions", None)
        if dimensions is not None:
            frame.configuration.dimensions = int(dimensions)
        frame.particles.N = int(source_particles.N)
        frame.particles.position = -np.asarray(source_particles.position, dtype=np.float64)

        for property_name in (
            "typeid",
            "types",
            "body",
            "diameter",
            "mass",
            "charge",
            "moment_inertia",
            "orientation",
            "velocity",
            "angmom",
            "image",
        ):
            copy_particle_property(frame.particles, source_particles, property_name)

        return frame


paths.flipped_initial_gsd.parent.mkdir(parents=True, exist_ok=True)
restart_frame = flipped_restart_frame(paths.in_gsd)
with gsd.hoomd.open(name=str(paths.flipped_initial_gsd), mode="w") as target:
    target.append(restart_frame)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=simulation.seed)
state = sim.create_state_from_gsd(filename=str(paths.flipped_initial_gsd))

integrator = hoomd.md.Integrator(dt=simulation.timestep)
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
        radius=analysis.cylinder_radius,
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
    filename=str(paths.flipped_gsd),
    trigger=hoomd.trigger.Periodic(simulation.flipped_trajectory_write_period),
    mode="wb",
    dynamic=["property", "particles/orientation"],
    logger=logger,
)
sim.operations.writers.append(gsd_writer)
print("starting sim")
sim.run(simulation.flipped_run_steps)
