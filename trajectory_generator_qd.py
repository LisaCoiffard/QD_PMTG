""" Generating a collection of trajectory generators using MAP-Elites"""

import sys, os
import argparse
import numpy as np
import pybullet as p

import map_elites.cvt as cvt_map_elites
import map_elites.grid as grid_map_elites
import map_elites.common as cm_map_elites
from environment import env_loader
from robot import a1

X_AMPLITUDE_SCALING = 0.1
Y_AMPLITURE_SCALING = 0.05
Z_AMPLITURE_SCALING = 0.05
BASE_FREQ_SCALING = 2.5
PHASE_OFFSETS_SCALING = 1


def calculate_simulation_offset(ref_leg, other_legs):
    t_ref = np.argmax(ref_leg)
    t_off_list = []

    for leg in other_legs:
        contacts = leg[t_ref:]
        t_off_list.append(np.argmax(contacts))
    return np.mean(t_off_list)


def normalise_phase_offset(ref_leg, offset):
    low_high_transitions = get_low_high_transitions(ref_leg)
    if len(low_high_transitions) < 2:
        return 0
    phases = np.diff(low_high_transitions)
    mean_phase = np.mean(phases)
    return offset/mean_phase


def get_low_high_transitions(ref_leg_contacts):

    low_high_transitions = []
    prev_contact = 0
    for i, contact in enumerate(ref_leg_contacts):
        if contact > prev_contact:
            low_high_transitions.append(i)
        prev_contact = contact
    return low_high_transitions


def simulate_tg_in_environment(params):
    env = env_loader.load("DIRECT",
                          environment='tg_simulation',
                          default_params=params)
    count = 0
    done = False
    positions = []
    phase_offsets = []

    fl_leg = []
    fr_leg = []
    rl_leg = []
    rr_leg = []

    while p.isConnected() and not done:
        pos, phase_offset, foot_contacts, done = env.step()
        # env.focus_camera_on_robot()
        count += 1

        positions.append(pos)
        phase_offsets.append(phase_offset)

        fr_leg.append(foot_contacts[0])
        fl_leg.append(foot_contacts[1])
        rr_leg.append(foot_contacts[2])
        rl_leg.append(foot_contacts[3])

        if count >= 700:
            p.disconnect()

    # compute fitness as straight line distance travelled from the initial position of the robot
    start_pos = a1.INIT_POSITION
    end_pos = positions[-1]
    f = end_pos[0] - start_pos[0]

    # compute descriptor dimension 1: actual phase offset of legs w.r.t FL leg
    mean_phase_offset = calculate_simulation_offset(fl_leg, [fr_leg, rr_leg, rl_leg])
    # NORMALISATION METHOD 2
    estimated_norm_factor = 5
    normalised_offset = mean_phase_offset/estimated_norm_factor
    # compute descriptor dimension 2: duty factor
    mean_duty_factor = np.mean([fr_leg, fl_leg, rr_leg, rl_leg])
    desc = [normalised_offset, mean_duty_factor]
    return f, desc


def scale_params(params):
    x_amplitude = params[0]*X_AMPLITUDE_SCALING
    y_amplitude = params[1]*Y_AMPLITURE_SCALING
    z_amplitude = params[2]*Z_AMPLITURE_SCALING

    speed = params[3]*BASE_FREQ_SCALING

    phase_offsets = params[-3:]*PHASE_OFFSETS_SCALING
    phase_offsets = np.insert(phase_offsets, 0, 0)

    return {'phase_offset': phase_offsets,
            'speed': speed,
            'x_amplitude': x_amplitude,
            'y_amplitude': y_amplitude,
            'z_amplitude': z_amplitude}


def get_tg_fitness_descriptor(params):
    scaled_params = scale_params(params)
    f, desc = simulate_tg_in_environment(scaled_params)
    return f, desc


if __name__=="__main__":

    px = cm_map_elites.default_params.copy()

    # use for debugging (no parallel evaluations)
    # px['batch_size'] = 1
    # px['random_init_batch'] = 1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    args = parser.parse_args()

    print(px)

    archive = grid_map_elites.compute(2, 7,
                                      get_tg_fitness_descriptor,
                                      (10, 50),
                                      max_evals=4e5,
                                      log_file=open(args.logdir + '/grid_tg.dat', 'w'),
                                      log_dir=args.logdir,
                                      params=px)
