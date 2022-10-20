import math
import numpy as np


class Task(object):
    def __init__(self,
                 trajectory_generator,
                 robot=None,
                 command_mode="fixed_dir",
                 max_terrain_height=None,
                 max_desired_velocity=1.2,
                 max_desired_yaw_rate=0.6):

        self._trajectory_generator = trajectory_generator
        self.command_mode = command_mode
        self.reset(robot, command_mode)

        self.max_terrain_height = max_terrain_height
        # print("MAX TERRAIN HEIGHT", self.max_terrain_height)
        self.max_desired_velocity = max_desired_velocity
        self.max_desired_yaw_rate = max_desired_yaw_rate
        self.foot_positions_history = np.zeros((3, 12))
        self.idx = -1

        self.stop_command = False

        self.desired_velocity = 0
        self.desired_yaw_rate = 0

    def reset(self, robot, command_mode):
        # print("RESET")
        self.robot = robot
        self.command_mode = command_mode
        self.command = Command(command_mode=self.command_mode, robot=self.robot)
        self.command.sample_command()
        # print("COMMAND", self.command.command)
        # print("GOAL POSITION", self.command._goal.goal_pos)
        self.desired_yaw_rate = 0
        self.desired_velocity = 0

    def set_desired_yaw_rate(self, yaw_rate):
        """Sets a new desired yaw rate"""
        self.desired_yaw_rate = yaw_rate

    def change_desired_yaw_rate(self, change):
        self.desired_yaw_rate += change
        self.desired_yaw_rate = min(max(self.desired_yaw_rate, -self.max_desired_yaw_rate), self.max_desired_yaw_rate)
        # print(self.desired_yaw_rate)

    def change_desired_forward_velocity(self, change):
        self.desired_velocity += change
        self.desired_velocity = min(max(self.desired_velocity, 0), self.max_desired_velocity)
        # print(self.desired_velocity)

    def set_desired_and_max_velocity(self, desired_velocity, max_velocity):
        self.desired_velocity = desired_velocity
        self.max_velocity = max_velocity

    def get_command(self):
        xy_command = self.command.command[:2] * self.desired_velocity
        yaw_command = self.command.command[-1] * self.desired_yaw_rate
        command = np.concatenate([xy_command.reshape(-1), np.array([yaw_command]).reshape(-1)], axis=0)
        return command

    def stop(self, bool):
        self.stop_command = bool

    def update_task(self, desired_velocity=None, desired_yaw_rate=None):
        self.command.update_command()
        if desired_velocity is not None:
            self.desired_velocity = desired_velocity
        if desired_yaw_rate is not None:
            self.desired_yaw_rate = desired_yaw_rate
        # print("COMMAND", self.get_command())
        # print(self.command.command)
        # print("GOAL POS", self.command._goal.goal_pos)

    def get_reward(self, measurements, action):
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
        command = self.command.command

        # print("DESIRED X VELOCITY:", command[0]*self.desired_velocity)
        # print("ACTUAL X VELOCITY:", measurements[0])
        # print("DESIRED Y VELOCITY:", command[1]*self.desired_velocity)
        # print("ACTUAL Y VELOCITY:", measurements[1])
        r_lv = 0.5 * math.exp(-15 * ((measurements[0] - command[0] * self.desired_velocity) ** 2)) + \
               0.5 * math.exp(-15 * ((measurements[1] - command[1] * self.desired_velocity) ** 2))
        # r_lv = 0.5 * math.exp(-30 * ((measurements[0] - command[0]*self.desired_velocity) ** 2)) + \
        #        0.5 * math.exp(-30 * ((measurements[1] - command[1]*self.desired_velocity) ** 2))

        # print("LINEAR VELOCITY REWARD TERM:", r_lv)

        # ANGULAR VELOCITY REWARD
        r_av = math.exp(-10 * ((measurements[4] - command[2]*self.desired_yaw_rate) ** 2))

        # print("YAW RATE", measurements[4])
        # print("COMMAND YAW RATE", self.desired_yaw_rate)
        # print("ANGULAR VELOCITY REWARD TERM:", r_av)

        # TARGET FOOT POSITION SMOOTHNESS
        self.idx = (self.idx + 1) % 3
        foot_positions = self._trajectory_generator.get_desired_foot_position().flatten()
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
        r_t = -self.robot.get_torques()
        # print("TORQUE PENALTY TERM", r_t)

        total_reward = 0.05 * r_lv + 0.05 * r_av + 0.025 * r_s + 0.005 * r_br + 0.01 * r_bp + 0.00002 * r_t
        # total_reward = 0.1 * r_lv + 0.05 * r_av + 0.025 * r_s + 0.005 * r_br + 0.01 * r_bp + 0.0002 * r_t

        return total_reward, r_lv, r_av, r_s, r_br, r_bp, r_t

    def check_default_terminal_condition(self):
        """Returns true if the robot is in a position that should terminate
        the episode, false otherwise"""

        roll, pitch, _ = self.robot.get_base_roll_pitch_yaw()
        pos = self.robot.get_base_position()
        # print(roll, pitch, pos[2])
        return abs(roll) > 1 or abs(pitch) > 1 or pos[2] < 0.10


class Command(object):
    def __init__(self, command_mode, robot):

        self.robot = robot
        self.command_mode = command_mode
        self._goal = Goal(robot=robot, command_mode=command_mode)

        self.command = np.zeros(3)
        self.command_dir = 0.0

    def update_command(self):
        if self.command_mode == "fixed_dir":
            self.command = np.array([1, 0, 0])

        else:
            # command is only updated if any of its components are non-zero
            if np.linalg.norm(self.command[:2]) != 0:
                x_dist = self._goal.goal_pos[0] - self.robot.get_base_position()[0]
                y_dist = self._goal.goal_pos[1] - self.robot.get_base_position()[1]

                # sample a new command only if the robot is less than 0.5 m from its goal
                if np.linalg.norm([x_dist, y_dist]) < 0.5:
                    self.sample_command()

                # otherwise compute new command from updated robot position
                else:
                    command_dir = np.arctan2(y_dist, x_dist)
                    self.command_dir = command_dir
                    yaw = self.robot.get_base_roll_pitch_yaw()[-1]
                    command_dir_body_frame = math.fmod(command_dir - yaw, math.pi * 2.0)

                    self.command[0] = np.cos(command_dir_body_frame)
                    self.command[1] = np.sin(command_dir_body_frame)

    def sample_command(self):
        if self.command_mode == "fixed_dir":
            self._goal.goal_pos = np.array([0, 0])
            self.command = np.array([1, 0, 0])

        else:
            self._goal.sample_goal()

            command_dir = np.arctan2(self._goal.goal_pos[1] - self.robot.get_base_position()[1],
                                     self._goal.goal_pos[0] - self.robot.get_base_position()[0])

            self.command_dir = command_dir
            yaw = self.robot.get_base_roll_pitch_yaw()[-1]
            command_dir_body_frame = math.fmod(command_dir - yaw, math.pi * 2.0)

            self.command[0] = np.cos(command_dir_body_frame)
            self.command[1] = np.sin(command_dir_body_frame)
            self.command[2] = 0.0

            if self.command_mode == "random":
                self.command[2] = 1 - 2 * np.random.randint(0, 1)
                self.command[2] *= np.random.uniform(0, 1)

            # print("Goal:", self._goal.goal_pos)
            # print("Command:", self.command)
            # print("Command direction", command_dir)


class Goal(object):
    def __init__(self, robot, command_mode):

        self.robot = robot
        self.goal_pos = np.zeros(2)

        self.command_mode = command_mode

        # TODO: change terrain dimensions - find out what these actually are
        self.terrain_size_x = 100
        self.terrain_size_y = 100

    def sample_goal(self):
        # print(self.robot)

        # if command mode is straight_x, only large change in the x goal coordinate
        if self.command_mode == "straight_x":
            self.goal_pos[1] = self.robot.get_base_position()[1] + self.terrain_size_y * 0.1 * np.random.uniform(-1, 1)
            self.goal_pos[0] = self.robot.get_base_position()[0] + self.terrain_size_x * 0.4
        # if command mode is straight_y, only large change in the y goal coordinate
        elif self.command_mode == "straight_y":
            self.goal_pos[0] = self.robot.get_base_position()[0] + self.terrain_size_x * 0.1 * np.random.uniform(-1, 1)
            self.goal_pos[1] = self.robot.get_base_position()[1] + self.terrain_size_y * 0.4

        # otherwise both coordinates are randomly assigned
        else:
            self.goal_pos[0] = self.robot.get_base_position()[0]
            self.goal_pos[1] = self.robot.get_base_position()[1]

            self.goal_pos[0] += self.terrain_size_x * 0.4 * np.random.uniform(-1, 1)
            self.goal_pos[1] += self.terrain_size_y * 0.4 * np.random.uniform(-1, 1)

        # clip goal to within terrain bounds
        self.goal_pos[0] = min([0.5 * self.terrain_size_x - 1, self.goal_pos[0]])
        self.goal_pos[0] = max([-0.5 * self.terrain_size_x + 1, self.goal_pos[0]])
        self.goal_pos[1] = min([0.5 * self.terrain_size_y - 1, self.goal_pos[1]])
        self.goal_pos[1] = max([-0.5 * self.terrain_size_y + 1, self.goal_pos[1]])




