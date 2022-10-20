""" Normalises behavioural descriptor outputs from the policy and outputs these as a dictionary of TG parameters."""

import numpy as np
from sklearn.neighbors import KDTree

from pmtg import trajectory_generator

X_AMPLITUDE_SCALING = 0.1
Y_AMPLITURE_SCALING = 0.05
Z_AMPLITURE_SCALING = 0.05
BASE_FREQ_SCALING = 2.5
PHASE_OFFSETS_SCALING = 1


class ParamPreProcessor:

    def __init__(self, archive):

        self._archive = archive
        self._fitness = self._archive[:, 0]
        self._centroid_list = self._archive[:, 1:3]
        self._min = np.min(self._centroid_list, 0)
        self._max = np.max(self._centroid_list, 0)
        # print(self._min, self._max)
        self._desc_list = self._archive[:, 3:5]
        self._x = self._archive[:, 5:]
        self._kdt = KDTree(self._centroid_list, leaf_size=30, metric='euclidean')

        self.selected_indices = []

    def _normalise_bd(self, policy_descriptor):
        # print(policy_descriptor)
        bd = policy_descriptor*(self._max-self._min) + self._min
        # print(bd)
        return bd

    def get_archive_index_from_policy(self, policy_descriptor):
        norm_policy_descriptor = self._normalise_bd(np.abs(policy_descriptor))
        return self._kdt.query([norm_policy_descriptor], k=1)[1][0][0]

    def preprocess_params(self, params):
        x_amplitude = params[0] * X_AMPLITUDE_SCALING
        y_amplitude = params[1] * Y_AMPLITURE_SCALING
        z_amplitude = params[2] * Z_AMPLITURE_SCALING

        speed = params[3] * BASE_FREQ_SCALING

        phase_offsets = params[-3:] * PHASE_OFFSETS_SCALING
        phase_offsets = np.insert(phase_offsets, 0, 0)

        return [x_amplitude, y_amplitude, z_amplitude, speed, phase_offsets]

    def make_param_dict(self, params):
        return {'phase_offset': params[-1],
                'speed': params[-2],
                'x_amplitude': params[0],
                'y_amplitude': params[1],
                'z_amplitude': params[2]}

    def get_param_dict_from_index(self, index):
        params = self._x[index]
        scaled_params = self.preprocess_params(params)
        return self.make_param_dict(scaled_params)

    def get_param_dict_from_descriptor(self, policy_descriptor):
        index = self.get_archive_index_from_policy(policy_descriptor)
        self.selected_indices.append(index)
        params = self._x[index]
        scaled_params = self.preprocess_params(params)
        return index, self.make_param_dict(scaled_params)

    def get_best_transition(self, delta, last_leg_phases, params):
        tg = trajectory_generator.TG(init_leg_phase_offsets=params['phase_offset'],
                                     default_speed=params['speed'],
                                     default_leg_amplitude_swing=params['x_amplitude'],
                                     default_leg_amplitude_lift=params['y_amplitude'],
                                     default_leg_amplitude_turn=params['z_amplitude'])

        prev_phases = [-1]
        new_phases = [0]
        min_phase_diff = np.inf
        best_init_leg_phase_offset = params['phase_offset']

        while prev_phases[0] < new_phases[0]:
            prev_phases, _ = tg.get_leg_phase_offsets()
            av_phase_diff = self._calculate_mean_phase_diff(prev_phases, last_leg_phases)
            if av_phase_diff < min_phase_diff:
                min_phase_diff = av_phase_diff
                best_init_leg_phase_offset = prev_phases
            tg.get_actions(delta, tg_params=None)
            new_phases, _ = tg.get_leg_phase_offsets()

        return best_init_leg_phase_offset

    def _calculate_mean_phase_diff(self, phases, phases_to_match):
        phase_diff = 0
        for i, phase in enumerate(phases):
            phase_diff += np.abs(phase - phases_to_match[i])
        mean_phase_diff = phase_diff/4
        return mean_phase_diff