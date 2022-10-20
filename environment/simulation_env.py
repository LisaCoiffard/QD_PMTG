"""TG simulation environment"""

import numpy as np
import math
from robot import a1
from utils import com_velocity_estimator

dt = 0.0005
ACTION_REPEAT = 20


class SimulationEnv:

    def __init__(self,
                 pybullet_client=None,
                 robot_class=None,
                 pmtg=None,
                 urdf_path=None,
                 scene_class=None,
                 simulation_time_step=dt):

        """
        Initializes the simulation environment for evaluating TGs.
        
        Args:
            pybullet_client: The instance of a pybullet client
            robot_class: The class of the robot. We prefer to pass the class
                instead of the instance because we might want to hard reset the
                environment by deleting it and creating it anew if it is too
                difficult to reset it.
            pmtg: an instance of the pmtg wrapper
            urdf_path: the path of the URDF of the robot used
            scene_class: class of the surrounding of the robot. If None it 
                simply initializes a normal plane.
        """

        self._pybullet_client = pybullet_client
        self._robot_class = robot_class
        self._pmtg = pmtg
        self._sim_time_step = dt
        self._sim_step_counter = 0
        self._env_step_counter = 0
        self._urdf_path = urdf_path
        self._scene_class = scene_class
        self._dt = simulation_time_step
        self._quadruped = None
        self._robot = None
        self._velocity_estimator = None
        self._vel_estimator_class = com_velocity_estimator.COMVelocityEstimator
        self._load()
        self._num_action_repeat = ACTION_REPEAT

        self._robot_observation_dims = 18
        self._velocity_dims = 3

    def _load_URDF(self):
        """Load the URDF

        Returns:
            _quadruped: the uid to give to the robot's class
        """
        return self._pybullet_client.loadURDF(self._urdf_path, a1.INIT_POSITION,
                                              a1._IDENTITY_ORIENTATION)

    def _load_instance(self):
        """Loads an instance of the robot from the class given

        Returns:
            _robot: the instance of the robot
        """
        # Uncomment the below to load the robot fixed in mid air (on-rack)
        # self._pybullet_client.createConstraint(self._quadruped, -1, -1, -1,
        #                                        self._pybullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
        #                                        [0, 0, 0.5], [0, 0, 0, 1])
        return self._robot_class(self._pybullet_client,
                                 self._quadruped, self._dt)

    def _load_scene(self):
        """Loads either a simple plane if the scene class is not defined, or the
        scene defined by the class. Be careful that when loading the standard
        plane it may have a different lateral friction coefficient than the scene
        which are normally used. This would cause the robot not to behave as
        expected"""
        if self._scene_class is None:
            plane_path = "data/plane.urdf"
            self._pybullet_client.loadURDF(plane_path)
        else:
            self._scene = self._scene_class(self._pybullet_client)
            # self._scene.create_scene()

    def _load(self):
        """Loads the environment and the robot
        
        Returns the new observation
        """

        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(self._sim_time_step)
        self._pybullet_client.setGravity(0, 0, -9.81)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=1)

        # Disables rendering to load faster
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 0)

        if (self._quadruped is not None) or (self._robot is not None):
            del self._quadruped
            del self._robot
            del self._velocity_estimator
        self._load_scene()
        self._quadruped = self._load_URDF()
        self._robot = self._load_instance()
        self._pmtg.reset()
        self._velocity_estimator = self._vel_estimator_class(self._robot)

        # Enables rendering after the loading
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 1)

        return self.check_default_terminal_condition()

    def _wait_for_rendering(self):
        """Sleep, otherwise the computation takes less time than real time
        To be finished. This is only used for real time rendering on GUI if the
        simulation is going too fast.
        """
        pass

    def hard_reset(self):
        """Destroys the simulation and resets everything, including the scene,
        the robot and all the objects. All the instances are destroyed and new
        ones are created from the classes."""

        self._env_step_counter = 0
        self._sim_step_counter = 0
        return self._load()

    def soft_reset(self):
        """Resets the robot pose, position and orientation in the simulation. 
        This might be faster than a hard reset, however if there dynamical 
        objects whose positions might have changed, it is better to hard_reset
        """

        self._robot.reset_pose_velocity_control()
        self._pybullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._quadruped,
            posObj=self._robot.GetDefaultInitPosition(),
            ornObj=self._robot.GetDeafaultInitOrientation()
        )

        self._sim_step_counter = 0
        self._env_step_counter = 0
        self._pmtg.reset()
        self._velocity_estimator.reset()
        # return self.get_state()

    def get_foot_position_in_hip_frame(self):
        """Returns x,y,z position of every foot in hip frame"""
        return self._robot.GetFootPositionsInBaseFrame()

    def get_foot_desired_position_in_hip_frame(self):
        """Returns desired x,y,z position of every foot in hip frame"""
        return self._pmtg.get_desired_foot_position()

    def check_default_terminal_condition(self):
        """Returns true if the robot is in a position that should terminate
        the episode, false otherwise"""

        roll, pitch, _ = self._robot.get_base_roll_pitch_yaw()
        pos = self._robot.get_base_position()
        # print(roll, pitch, pos[2])
        return abs(roll) > 1 or abs(pitch) > 1 or pos[2] < 0.10

    def get_robot_position(self):
        return self._robot.get_base_position()

    def focus_camera_on_robot(self):
        """Sets the camera to follow the robot. Has to be called at every time
        step. Good for taking videos on GUI connection"""
        self._pybullet_client.resetDebugVisualizerCamera(1.0, 0, -30,
                                                         self._robot.get_base_position())

    def start_recording(self):
        """Start recording a video. Only available on GUI connection. It slows
        down the simluation *significantly*. You need to install ffmpeg"""
        self._pybullet_client.startStateLogging(
            loggingType=self._pybullet_client.STATE_LOGGING_VIDEO_MP4,
            fileName="video.mp4")

    def stop_recording(self):
        """Stop the video recording"""
        self._pybullet_client.stopStateLogging(
            loggingType=self._pybullet_client.STATE_LOGGING_VIDEO_MP4,
            fileName="video.mp4")

    def step(self, action=None):
        """Step forward the simulation, given an action of the robot.

        Args:
            action: The x,y,z coordinates of each foot link.
            
        Returns:
            state: a dictionary where the keys are
                the sensor names and the values are the sensor readings.
            reward: the reward obtained by the robot. For the moment it will be
                Null since we are not doing RL but just testing out the env
            done: whether or not the episode terminated.
        """

        self._sim_step_counter += self._sim_time_step * self._num_action_repeat
        self._env_step_counter += self._num_action_repeat

        link_positions, _ = self._pmtg.step(time=self._sim_step_counter,
                                         action=action)

        for _ in range(self._num_action_repeat):
            self._robot.inverse_kinematics_action(link_positions)
        self._velocity_estimator.update()

        done = self.check_default_terminal_condition()
        pos = self.get_robot_position()
        phase_offsets = self._pmtg.get_leg_phase_offsets()
        foot_contacts = self._robot.get_foot_contacts()

        return pos, phase_offsets, foot_contacts, done
