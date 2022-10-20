"""Adapted from Davide Paglieri repository available at: https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2020-2021/davide_paglieri/control_pmtg
PMTG wrapper"""
# Special thanks to Atil Iscen for helping with the implementation
# Part of the implementation is adapted from PyBullet.

import numpy as np
from pmtg import cyclic_integrator
from pmtg import trajectory_generator
from robot import a1
from utils import action_filter, param_preprocessor

# These gaits are for FL, RL, FR, RR.
_GAIT_PHASE_MAP = {
    "walk": [0, 0.25, 0.5, 0.75],
    "trot": [0, 0.5, 0.5, 0],
    "bound": [0, 0.5, 0, 0.5],
    "pace": [0, 0, 0.5, 0.5],
    "pronk": [0, 0, 0, 0]
}

residual_ranges = np.array([0.1, 0.1, 0.05,
                            0.1, 0.1, 0.05,
                            0.1, 0.1, 0.05,
                            0.1, 0.1, 0.05, ])


class PMTG():
    """PMTG wrapper's only role is to decompose the actions,
    call the integrator to progress the phase and call trajectory 
    generator to obtain motor positions based on the new phase. 
    It also serves as the main hub to do decouplings and assigning 
    legs to different integrators if you prefer to have the legs 
    decoupled.
    """

    def __init__(self,
                 residual_range=residual_ranges,
                 init_leg_phase_offsets=None,
                 init_gait=None,
                 action_filter_enable=False,
                 action_filter_order=1,
                 action_filter_low_cut=0,
                 action_filter_high_cut=3.0,
                 action_filter_initialize=True,
                 archive_filename='',
                 tg_select=None,
                 default_tg_params=None):
        """Initialzes the wrapped env.

        Args:
            residual_range: The upper limit for the residual actions that adds to
                the leg motion. By default it is 0.1 for x,y residuals, and 0.05 for
                z residuals.
            init_leg_phase_offsets: The initial phases of the legs. A list of 4
                variables within [0,1). The order is front-left, rear-left,
                front-right and rear-right.
            init_gait: The initial gait that sets the starting phase difference
                between the legs. Overrides the arg init_phase_offsets. Has to be
                "walk", "trot", "bound" or "pronk". Used in vizier search.
            action_filter_enable: Use a butterworth filter for the output of the
                PMTG actions (before conversion to leg swing-extend model). It
                forces smoother behaviors depending on the parameters used.
            action_filter_order: The order for the action_filter (1 by default).
            action_filter_low_cut: The cut for the lower frequencies (0 by default).
            action_filter_high_cut: The cut for the higher frequencies
                (3 by default).
            action_filter_initialize: If the action filter should be initialized
                when the first action is taken. If enabled, the filter does not
                affect action value the first time it is called and fills the
                history with that value.
            archive_filename: Name of file containing collection of TGs.
            tg_select: If True, the policy can select a TG, otherwise it uses the same one that is instantiated throughout.
            default_tg_params: Used to simulate TG parameters for MAP-Elites archive generation

        Raises:
            ValueError if the controller does not implement get_action and
            get_observation.
        """

        # Policy actions are the x,y,z residuals of each foot in the hip frame
        self._num_actions = 12
        self._residual_range = residual_range

        self._use_archive = True if archive_filename else False
        self._enable_tg_select = tg_select

        if not default_tg_params:
            if self._use_archive and tg_select:
                self._archive = np.loadtxt(archive_filename)
                self._tg_param_preprocessor = param_preprocessor.ParamPreProcessor(self._archive)
            elif self._use_archive:
                self._archive = np.loadtxt(archive_filename)
                index = 165
                # index = np.random.randint(0, len(self._archive))
                self._tg_param_preprocessor = param_preprocessor.ParamPreProcessor(self._archive)
                default_params = self._tg_param_preprocessor.get_param_dict_from_index(index)
        else:
            default_params = default_tg_params

        # Phase of the TG can be initialised by either:
        # 1. specifying the exact phase offsets e.g. [0, 0.5, 0.5, 0]
        # 2. specifying the gait from a gait library which has fixed phases e.g. "trot"
        # 3. if neither are specified, "trot" gait is used by default
        if init_leg_phase_offsets is not None:
            init_leg_phase_offsets = init_leg_phase_offsets
        elif init_gait:
            if init_gait in _GAIT_PHASE_MAP:
                init_leg_phase_offsets = _GAIT_PHASE_MAP[init_gait]
        else:
            init_leg_phase_offsets = _GAIT_PHASE_MAP["trot"]

        # Initialise the TG:
        # 1. If archive specified but policy not selecting TGs, use parameters of particular TG from the archive
        # throughout
        # 2. If archive specified and policy selecting TGs initialise to default TG and change these later in the step
        # method
        # 3. If default_tg_params initialise TG from input genotype (used to generate MAP-Elites archive)
        if (self._use_archive and not self._enable_tg_select) or default_tg_params is not None:
            self._trajectory_generator = trajectory_generator.TG(
                init_leg_phase_offsets=default_params["phase_offset"],
                default_speed=default_params["speed"],
                default_leg_amplitude_swing=default_params["x_amplitude"],
                default_leg_amplitude_lift=default_params["z_amplitude"],
                default_leg_amplitude_turn=default_params["y_amplitude"])
        else:
            self._trajectory_generator = trajectory_generator.TG(
                init_leg_phase_offsets=init_leg_phase_offsets)

        self._extend_obs_space()
        # Extend action dimension to include TG parameters - used in creation of action filter
        action_dim = self._extend_action_space()
        # Initialise action filter
        self._action_filter_enable = action_filter_enable
        if self._action_filter_enable:
            self._action_filter_initialize = action_filter_initialize
            self._action_filter_order = action_filter_order
            self._action_filter_low_cut = action_filter_low_cut
            self._action_filter_high_cut = action_filter_high_cut
            self._action_filter = self._build_action_filter(action_dim)

        self.reset()

    def _extend_obs_space(self):
        """Extend observation space to include pmtg phase variables."""
        pass

    def _extend_action_space(self):
        """Extend the action space to include pmtg parameters."""
        return 17

    def get_phase(self):
        """Returns the phase of the trajectory generator"""
        return self._trajectory_generator.get_state()

    def get_leg_phase_offsets(self):
        """Returns the actual phase offsets of the legs relative to FL leg.
        These are returned in order [FL, RL, FR, RR]."""
        return self._trajectory_generator.get_leg_phase_offsets()[1]

    def get_leg_phases(self):
        """Returns the phase of each leg"""
        phases = []
        for leg in self._trajectory_generator._legs:
            phases.append(leg.phase)
        phases = np.array([phases[2],
                           phases[0],
                           phases[3],
                           phases[1]])
        return phases

    def _get_observation_bounds(self):
        """Get the bounds of the observation added from the trajectory generator

        Returns:
            lower_bounds: Lower bounds for observations
            upper_bounds: Upper bounds for observations
        """
        lower_bounds = self._trajectory_generator.get_state_lower_bounds()
        upper_bounds = self._trajectory_generator.get_state_upper_bounds()
        return lower_bounds, upper_bounds

    def step(self, time, action=None):
        """Make a step of the pmtg wrapper

        Args:
            action: a numpy array composed of the policy residuals and the
                time_multiplier, intensity, walking_height, swing_stance_ratio
                used by the Trajectory Generator
            time: time since reset in seconds in the environment

        Returns:
            link_positions: a numpy array composed of the positions of the link
                feet.
            index: an integer for the index of the current TG in the archive, used to compute TG switch reward term
        """

        delta_real_time = time - self._last_real_time
        self._last_real_time = time

        if action is not None:

            # Apply action filter
            if self._action_filter_enable:
                action = self._filter_action(action)

            if self._enable_tg_select:
                last_phases, _ = self._trajectory_generator.get_leg_phase_offsets()
                index, default_tg_params = self._tg_param_preprocessor.get_param_dict_from_descriptor(action[-2:])
                if self.prev_phase[0] < last_phases[0]:
                    pass
                else:
                    if self.prev_index != index:
                        last_phases, _ = self._trajectory_generator.get_leg_phase_offsets()
                        init_leg_phase_offsets = self._tg_param_preprocessor.get_best_transition(delta_real_time,
                                                                                                 last_phases,
                                                                                                 default_tg_params)
                        self._trajectory_generator = trajectory_generator.TG(
                            init_leg_phase_offsets=default_tg_params["phase_offset"],
                            default_speed=default_tg_params["speed"],
                            default_leg_amplitude_swing=default_tg_params["x_amplitude"],
                            default_leg_amplitude_lift=default_tg_params["z_amplitude"],
                            default_leg_amplitude_turn=default_tg_params["y_amplitude"])

                        self.prev_index = index

                action = action[:-2]
                self.prev_phase = last_phases

            # Retrieve policy residual actions and clip to residual range
            residuals = action[0:self._num_actions] * self._residual_range
            # Calculate trajectory generator's output based on the TG params (rest of the actions from the policy)
            action_tg = self._trajectory_generator.get_actions(
                delta_real_time, action[self._num_actions:])
            # Robot actions as the sum of TG actions and policy residuals
            link_positions = action_tg + residuals

        else:
            link_positions = self._trajectory_generator.get_actions(
                delta_real_time, None)

        # Re-order for leg order to match outputs of joint angles
        link_positions = np.array([link_positions[6:9],
                                   link_positions[0:3],
                                   link_positions[9:12],
                                   link_positions[3:6]])
        return link_positions, self.prev_index

    def reset(self):
        """Reset the Trajectory Generators, PMTG's parameters and action filter.
        """
        self._last_real_time = 0
        self.prev_index = -1
        self.prev_phase = [-1]
        self._trajectory_generator.reset()

        if self._action_filter_enable:
            self._reset_action_filter()

    def enable_tg(self):
        self._trajectory_generator._stop_command = False

    def disable_tg(self):
        self._trajectory_generator._stop_command = True

    def get_num_actions(self):
        """Number of action dimensions outputted by the policy. If the policy selects TGs add 2 dimensions for
        behavioural descriptor."""
        if self._enable_tg_select:
            # add + 2 if selecting TG with policy
            return self._trajectory_generator.num_integrators + 12 + 2
        else:
            return self._trajectory_generator.num_integrators + 12

    def get_num_states(self):
        """Number of TG observation dimensions used as input to the policy."""
        return self._trajectory_generator.num_integrators * 2

    def _build_action_filter(self, num_joints):
        """Creates and returns a Butterworth filter for the actions. This is
        done so that the action frequency response is as flat as possible.
        This should in theory smooth the action of the robot, and prevent joint
        malfunctions in the real world robot. To be experimented whether it
        improves performance in simulation or not.

        Args:
            num_joints: DoF of the robot

        Returns:
            action_filter: The butterworth filter
        """
        order = self._action_filter_order
        low_cut = self._action_filter_low_cut
        high_cut = self._action_filter_high_cut
        sampling_rate = 1 / (0.01)
        a_filter = action_filter.ActionFilterButter([low_cut], [high_cut],
                                                    sampling_rate, order,
                                                    num_joints)
        return a_filter

    def _reset_action_filter(self):
        """Resets the filter buffer"""
        self._action_filter.reset()
        self._action_filter_empty = True
        return

    def _filter_action(self, action):
        """Applies the filter to the action. If the filter's buffer is empty
        it initializes it"""
        if self._action_filter_empty and self._action_filter_initialize:
            # If initialize is selected and it is the first time filter is 
            # called, fill the buffer with that action so that it starts from 
            # that value instead of zero(s).
            init_action = np.array(action).reshape(len(action), 1)
            self._action_filter.init_history(init_action)
            self._action_filter_empty = False
        filtered_action = self._action_filter.filter(np.array(action))
        return filtered_action

    def get_desired_foot_position(self):
        positions = self._trajectory_generator.get_leg_desired_positions()
        return (np.array([positions[6:9],
                          positions[0:3],
                          positions[9:12],
                          positions[3:6]])).reshape(4, 3)

    def increase_speed(self, delta):
        self._trajectory_generator.increase_speed(delta)
