# Deep Reinforcement Learning for Dynamic Robot Locomotion 

This repository contains implementations for the project "Deep Reinforcement Learning for Dynamic Robot Locomotion", by
Lisa Coiffard (2022).

This provides implementations for the training and visualisations of trained controllers presented in the main report.

## Installation

To clone this repository run the following command in terminal:
``git clone --recurse-submodules https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2021-2022/lisa_coiffard/qd_pmtg``

## Dependencies
The trained controllers can be visualised in simulation by installing the following dependencies:
`pip install pybullet gym absl-py numpy opensimplex==0.3 matplotlib seaborn sklearn`

## Examples

Supplementary videos links can be found in report's appendix A.

To visualise the generated archive of TGs run `python plot_map_elites/plot_2d_map.py centroids_500.dat archive_400000.dat`

To visualise the trained controller for the specialist agent on flat terrains first assign the index 178 to line 93 of
the `pmtg_wrapped.py` script to select the TG it was trained on, then run `python visualise_flat_terrain.py 
--archive=archive_400000.dat --filename=policies/specialist_flat.npz`

To visualise the trained controller for the generalist agent on flat terrains run `visualise_flat_terrain.py 
--archive=archive_400000.dat --filename=policies/generalist_flat.npz --tg_select=1`

To visualise the trained controller for the specialist agent on flat terrains first assign the index 178 to line 93 of
the `pmtg_wrapped.py` script to select the TG it was trained on, then run `python visualise_domain_randomisation.py 
--archive=archive_400000.dat --filename=policies/specialist_terrains.npz`

To visualise the trained controller for the generalist agent on flat terrains first assign the index 178 to line 93 of
the `pmtg_wrapped.py` script to select the TG it was trained on, then run `python visualise_domain_randomisation.py 
--archive=archive_400000.dat --filename=policies/generalist_terrains.npz --tg_select=1`

## Basic usage

To train the controllers according to the methods presented in the main report, run the chosen experiment on the HPC
by uncommenting the appropriate training file command in the `singularity.def` file. Push the changes. Change directory
to the `singularity` folder and run the command:

`./build_final_image`

Copy to the final image to the HPC with the command:

`spc final_qd_pmtg_XXX.sif user@login.hpc.ic.ac.uk:`

(replace user with your user name and select the appropriate image name).
You can now create a .job script and submit your job to train your agent.