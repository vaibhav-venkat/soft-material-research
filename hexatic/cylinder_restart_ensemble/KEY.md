# Cylinder Restart Ensemble Key

All scripts restart from the last frame of `hexatic/output/cylinder/trajectory_cylinder.gsd`.
Each script refuses to overwrite an existing initial-condition GSD or trajectory GSD.
Outputs are written to `hexatic/output/cylinder/restart_ensemble/`.

| Letter | Script | Output | Meaning |
| --- | --- | --- | --- |
| A | `run_A_exact_restart.py` | `trajectory_cylinder_A_i.gsd` | Exact restart ensemble. Positions, orientations, and `U0` unchanged; randomized seed per replica. Set `N_REPLICAS` in the script. |
| B | `run_B_u0_half.py` | `trajectory_cylinder_B.gsd` | Same restart frame with active propulsion `U0 = 0.5 * U0_original`. |
| C | `run_C_random_orientations.py` | `trajectory_cylinder_C.gsd` | Positions unchanged; active orientations randomized uniformly. |
| D | `run_D_shuffle_orientations.py` | `trajectory_cylinder_D.gsd` | Positions unchanged; original orientations shuffled among particles. |
| F | `run_F_zero_px.py` | `trajectory_cylinder_F.gsd` | Positions unchanged; orientations adjusted so mean axial polarization `P_x = 0`. |
| G | `run_G_u0_double.py` | `trajectory_cylinder_G.gsd` | Same restart frame with active propulsion `U0 = 2 * U0_original`. |
| H | `run_H_consistent_axial_mirror.py` | `trajectory_cylinder_H.gsd` | Consistent axial mirror: `x -> -x`, `p_x -> -p_x`. |
| I | `run_I_orientation_axial_flip.py` | `trajectory_cylinder_I.gsd` | Orientation-only axial flip: positions unchanged, `p_x -> -p_x`. |
| J | `run_J_theta_mirror.py` | `trajectory_cylinder_J.gsd` | Theta mirror with `theta = atan2(z, y)`: `z -> -z`, `p_z -> -p_z`. |
| S | `run_S_passive_u0_zero.py` | `trajectory_cylinder_S.gsd` | Passive relaxation: active propulsion set to `U0 = 0`. |
| T | `run_T_reverse_activity.py` | `trajectory_cylinder_T.gsd` | Reverse activity: active propulsion set to `U0 = -U0_original`. |

`K` is not a separate script because it was clarified to be the same as the `J` theta mirror with `x` unchanged.

For all activity-strength tests, only active propulsion is changed. LJ and wall interaction strengths remain fixed at the original positive `U0_original` value so each condition changes one intended variable.
