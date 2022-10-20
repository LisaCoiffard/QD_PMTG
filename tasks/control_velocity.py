"""Adapted from Davide Paglieri repository available at: https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2020-2021/davide_paglieri/control_pmtg
A simple task where the reward is based on how close the robot velociy is
to the desired velocity"""

import math
import numpy as np


class Task(object):
    """A simple task where the reward is based on how close the robot velociy is
    to the desired velocity"""

    def __init__(self,
                 robot=None,
                 max_desired_velocity_x=1.8,
                 max_desired_velocity_y=0.2,
                 max_desired_yaw_rate=0.6):
        """Init of the Task

        Args:
            desired_velocity: velocity that will achieve the maximum reward
            max_desired_velocity: maximum desired velocity
        """
        self.robot = robot

        self.desired_velocity_x = 0.0
        self.max_desired_velocity_x = max_desired_velocity_x
        self.desired_velocity_y = 0.0
        self.max_desired_velocity_y = max_desired_velocity_y
        self.desired_yaw_rate = 0.0
        self.stop_command = False
        self._direction = 0

        self.foot_positions_history = np.zeros((3, 12))
        self.idx = -1

    def reset(self, robot):
        """Initializes a new instance of the robot and resets the
        desired velocity"""
        self.robot = robot
        self.desired_velocity_x = 0.0
        self.desired_velocity_y = 0.0
        self.desired_yaw_rate = 0.0

        self.prev_index = -1

        self.get_desired_linear_velocity()

    def change_speed_linearly(self, axis="x", sign=1):
        """Linearly changes the desired speed along x or y axes until it reaches the maximum
        desired speed or 0 m/s

        Args:
            axis: "x" or "y" according to which axis along which to change speed
            sign: +1 or -1, the desired direction of the change
        """
        if axis == "x":
            if (sign == 1 and (
                    0.0 <= self.desired_velocity_x < self.max_desired_velocity_x)):
                self.desired_velocity_x += 0.002
            elif ((sign == -1) and (self.desired_velocity_x >= 0.0)):
                self.desired_velocity_x -= 0.002
        elif axis == "y":
            if (sign == 1 and (
                    0.0 <= self.desired_velocity_y < self.max_desired_velocity_y)):
                self.desired_velocity_y += 0.002
            elif ((sign == -1) and (self.desired_velocity_y >
                                    -self.max_desired_velocity_y)):
                self.desired_velocity_y -= 0.002
        else:
            raise ("Wrong axis input")

        self.get_desired_linear_velocity()

    def set_desired_yaw_rate(self, yaw_rate):
        print()
        """Sets a new desired yaw rate"""
        self.desired_yaw_rate = yaw_rate

    def change_yaw_rate_linearly(self, sign=1):
        if sign == 1 and (self.desired_yaw_rate < 0.5):
            self.desired_yaw_rate += 0.001
        if sign == -1 and (self.desired_yaw_rate > -0.5):
            self.desired_yaw_rate -= 0.001

    def get_desired_linear_velocity(self):
        self.desired_lv = np.linalg.norm([self.desired_velocity_x, self.desired_velocity_y])

    def set_desired_and_max_velocity(self, desired_velocity, max_velocity):
        self.desired_velocity = desired_velocity
        self.max_velocity = max_velocity

    def get_command(self):
        """Get desired velocity of the robot CoM and yaw rate"""
        return (self.desired_velocity_x, self.desired_velocity_y,
                self.desired_yaw_rate)

    def stop(self, bool):
        self.stop_command = bool

    def set_angle(self, angle):
        self._direction = angle

    def get_reward(self, measurements, actions, tg_index=None):
        """ Get reward for the current time step.

        Args:
            measurements: a tuple of robot measurements in order, x velocity, y velocity, roll rate,
            pitch rate and yaw rate.

        Returns:
            total_reward: reward for the robot's step calculated as:
                totat_reward = 0.05*r_lv + 0.05*r_av + 0.025*r_s + 0.005*r_br + 0.01*r_bp + 0.0002*r_t
            r_lv: linear velocity reward term
            r_av: angular velocity reward term
            r_s: smoothness reward term (as a penalty)
            r_br: base roll reward term
            r_bp: base pitch reward term
            r_t: torque reward term (as a penalty)
        """

        # LINEAR VELOCITY REWARD
        x_hat = math.cos(self._direction)
        y_hat = math.sin(self._direction)
        command = np.vstack([x_hat, y_hat])

        r_lv = 0.5 * math.exp(-15 * ((measurements[0] - self.desired_velocity_x) ** 2)) + \
               0.5 * math.exp(-15 * ((measurements[1] - self.desired_velocity_y) ** 2))

        # print("LINEAR VELOCITY REWARD TERM:", r_lv)

        # ANGULAR VELOCITY REWARD
        r_av = math.exp(-10 * ((measurements[4] - self.desired_yaw_rate) ** 2))

        # print("YAW RATE", measurements[4])
        # print("COMMAND YAW RATE", self.desired_yaw_rate)
        # print("ANGULAR VELOCITY REWARD TERM:", r_av)

        # TARGET FOOT POSITION SMOOTHNESS
        self.idx = (self.idx + 1) % 3
        foot_positions = actions.flatten()
        self.foot_positions_history[self.idx, :] = foot_positions
        # print("FOOT POS HISTORY", self.foot_positions_history)
        r_s = -np.linalg.norm(self.foot_positions_history[self.idx, :] -
                              2 * self.foot_positions_history[self.idx - 1, :] +
                              self.foot_positions_history[self.idx - 2, :], ord=2)
        # print("SMOOTHNESS REWARD TERM:", r_s)

        # BASE MOTION
        r_br = math.exp(-2 * (abs(measurements[2])))
        r_bp = math.exp(-2 * (abs(measurements[3])))
        # print("ROLL MEASUREMENT", measurements[2])
        # print("BASE MOTION ROLL REWARD:", r_br)
        # print("PITCH MEASUREMENT", measurements[3])
        # print("BASE MOTION PITCH REWARD:", r_bp)

        # TORQUE PENALTY
        # print("MOTOR TORQUES", self.robot.get_applied_motor_torque())
        # r_t = -self.robot.get_torques()
        r_t = -np.sum(np.absolute(self.robot.get_applied_motor_torque()))
        # print("TORQUE PENALTY TERM", r_t)

        # print(tg_index)
        if tg_index is not None:
            if tg_index != self.prev_index and self.prev_index != -1:
                r_c = -1
            else:
                r_c = 0
            self.prev_index = tg_index
        else:
            r_c = 0
        # print("SWITCH PENALTY TERM:", r_c)

        total_reward = 0.05 * r_lv + 0.05 * r_av + 0.025 * r_s + 0.005 * r_br + 0.01 * r_bp + 0.0002 * r_t + 0.01*r_c
        # total_reward = 0.05 * r_lv + 0.05 * r_av + 0.025 * r_s + 0.005 * r_br + 0.01 * r_bp + 0.0002 * r_t

        return total_reward, r_lv, r_av, r_s, r_br, r_bp, r_t

    def check_default_terminal_condition(self):
        """Returns true if the robot is in a position that should terminate
        the episode, false otherwise"""

        roll, pitch, _ = self.robot.get_base_roll_pitch_yaw()
        pos = self.robot.get_base_position()
        # print(roll, pitch, pos[2])
        return abs(roll) > 1 or abs(pitch) > 1 or pos[2] < 0.10
