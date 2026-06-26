# Architecture Overview

## 1. Project Structure

.
├── hexatic
│   ├── active_matter_cylinder # calculate active_matter_fields such as polarization
│   │   ├── cartesian
│   │   ├── fields
│   │   └── shear
│   ├── analysis # mostly deprecated -> used to analyze hexatic order
│   ├── chirality # calculate chirality across CCM, chi, translation
│   ├── constants # useful constants contained for simulations/analysis
│   ├── cylinder_dynamics # dynamic values like velocity for static cylinder
│   ├── cylinder_restart_ensemble # deprecated, flipped *.gsd
│   ├── lagged_prediction # predictive lagging, using scikit
│   ├── multiple_sim_analysis # current focus, analyze across multiple *.gsd
│   │   ├── disclination_order_fields # current focus, on disclination behaviors
│   │   └── output
│   │       ├── fits
│   │       ├── npz
│   │       └── plots
│   ├── output
│   │   ├── cylinder
│   │   │   ├── chirality_disclinations
│   │   │   ├── images
│   │   │   └── restart_ensemble
│   │   ├── images
│   │   │   ├── active
│   │   │   ├── chirality
│   │   │   ├── disc
│   │   │   └── lagged_prediction
│   │   └── sphere
│   └── radii_analysis # code to generate the multiple simulations
│       ├── gsd
│       ├── hexatic_output
│       ├── initial
│       ├── logs
│       ├── metadata
│       └── npz_fields
├── logs 
└── tests



## 2. High-Level System Diagram
 
[User] <--> [Plots/fits dynamics and fields] <--> [Runs multiple_sim_analyis] <--> [GSD, NPZ, and archive files]

## 3. Unique Components

- Numba: aim to use this extensively unless it increases too much complexity.
- Scipy: aim to use this instead of numpy for more mathematically heavy data structures
- Pixi:
  - env_name: sap, default
  - command: `pixi shell`
  - contained in [pixi.toml](~/structures_and_functions/pixi.toml)
