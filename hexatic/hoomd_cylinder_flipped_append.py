import hoomd

if __package__:
    from hexatic.constants import cylinder
else:
    from constants import cylinder

paths = cylinder.PATHS
analysis = cylinder.ANALYSIS
simulation = cylinder.SIMULATION

RUN_STEPS = int(4e7)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=simulation.seed)
state = sim.create_state_from_gsd(filename=str(paths.flipped_gsd))

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
    trigger=hoomd.trigger.Periodic(simulation.trajectory_write_period),
    mode="ab",
    dynamic=["property", "particles/orientation"],
    logger=logger,
)
sim.operations.writers.append(gsd_writer)
print(f"appending sim for {RUN_STEPS} steps, {simulation.trajectory_write_period} write_period")
sim.run(RUN_STEPS)
