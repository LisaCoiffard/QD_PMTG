"""Adapted from Davide Paglieri repository available at: https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2020-2021/davide_paglieri/control_pmtg
Locomotion environment"""

import numpy as np
import math
from robot import a1
from utils import com_velocity_estimator

dt = 0.0005
ACTION_REPEAT = 20


class LocomotionEnv:

    def __init__(self,
                 pybullet_client=None,
                 robot_class=None,
                 pmtg=None,
                 urdf_path=None,
                 scene_class=None,
                 max_terrain_height=None,
                 task=None,
                 command_mode=None,
                 simulation_time_step=dt):

        """Initializes the locomotion environment
        
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
            task: An istance of a task class, it contains the reward function 
                and terminal conditions of each episode.
        """

        self._pybullet_client = pybullet_client
        self._robot_class = robot_class
        self._pmtg = pmtg
        self._task = task
        self._sim_time_step = dt
        self._sim_step_counter = 0
        self._env_step_counter = 0
        self._urdf_path = urdf_path
        self._scene_class = scene_class
        self._max_terrain_height = max_terrain_height
        self._dt = simulation_time_step
        self._quadruped = None
        self._robot = None
        self._command_mode = command_mode
        self._velocity_estimator = None
        self._vel_estimator_class = com_velocity_estimator.COMVelocityEstimator
        self._command_mode = command_mode
        self._load()
        if self._command_mode:
            self._task.reset(self._robot, self._command_mode)
        else:
            self._task.reset(self._robot)
        self._num_action_repeat = ACTION_REPEAT

        self._robot_observation_dims = 18
        self._velocity_dims = 3

    def _load_URDF(self, init_pos=None):
        """Load the URDF

        Args:
            init_pos: The initial position of the robot in the world frame.
                If not specified, the robot is initialised at position [0, 0, 0.32]

        Returns:
            _quadruped: the uid to give to the robot's class
        """
        if init_pos:
            return self._pybullet_client.loadURDF(self._urdf_path, init_pos,
                                                  a1._IDENTITY_ORIENTATION)
        else:
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

    def _load_scene(self, heightfield=None, is_slope=False):
        """Loads either a simple plane if the scene class is not defined, or the
        scene defined by the class. Be careful that when loading the standard
        plane it may have a different lateral friction coefficient than the scene
        which are normally used. This would cause the robot not to behave as
        expected

        Args:
            heightfield: Heightfield to create a terrain that is not a flat plane.
                If not specified, the terrain is instantiated as a flat plane.
            is_slope: If True, the terrain is initialised at a different position than other terrains to account
                for the robot's positioning at initialisation.
        """

        if heightfield is None:
            heightfield = np.zeros(128 * 128)
            terrain_mass = 0
            terrain_visual_shape_id = -1
            terrain_position = [0, 0, 0]
            terrain_orientation = [0, 0, 0, 1]
            self.terrainShape = self._pybullet_client.createCollisionShape(
                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[1, 1, 1],
                numHeightfieldRows=128,
                numHeightfieldColumns=128,
                heightfieldData=heightfield,
                heightfieldTextureScaling=128)

        else:
            size = int(math.sqrt(heightfield.shape[0]))
            self.terrainShape = self._pybullet_client.createCollisionShape(
                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[0.025, 0.025, 1],
                # meshScale=[0.05, 0.05, 1],
                numHeightfieldRows=size,
                numHeightfieldColumns=size,
                heightfieldData=heightfield,
                heightfieldTextureScaling=128)
        self.terrain = self._pybullet_client.createMultiBody(0, self.terrainShape)

        # NICER TEXTURE FOR RECORDING
        texUid = self._pybullet_client.loadTexture("/data/grid1.png")
        self._pybullet_client.changeVisualShape(self.terrain,
                                                -1,
                                                rgbaColor=[0.9, 0.9, 0.9, 1])
        self._pybullet_client.changeVisualShape(self.terrain,
                                                -1,
                                                textureUniqueId=texUid)

        # FRICTION AND COLLISION MARGIN
        if is_slope:
            init_pos = [0,0, (heightfield[-1]-heightfield[0])/2]
        else:
            init_pos = [0,0,0]

        self._pybullet_client.resetBasePositionAndOrientation(self.terrain,
                                                              init_pos,
                                                              [0, 0, 0, 1])
        self._pybullet_client.changeDynamics(self.terrainShape,
                                             -1,
                                             lateralFriction=1)
        self._pybullet_client.changeDynamics(self.terrainShape,
                                             -1,
                                             collisionMargin=0.01)

    def _load(self, heightfield=None, terrain_enc=None, is_slope=False):
        """Loads the environment and the robot

        Args:
            heightfield: Load terrains other than a flat plane when loading the scene.
            terrain_enc: When training on varied terrains to add as part of state information.
            init_pos: Change the initial robot position.
            is_slope: Initialises slope terrain at a different initial position if True.
        
        Returns the new observation
        """

        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(self._sim_time_step)
        self._pybullet_client.setGravity(0, 0, -9.81)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=1,
                                                        allowedCcdPenetration=0.0)

        # Disables rendering to load faster
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 0)

        if (self._quadruped is not None) or (self._robot is not None):
            del self._quadruped
            del self._robot
            del self._velocity_estimator
        self._load_scene(heightfield, is_slope)
        if is_slope:
            self._quadruped = self._load_URDF(init_pos=[-2.5, 0, 0.32])
        else:
            self._quadruped = self._load_URDF()
        self._robot = self._load_instance()
        self._pmtg.reset()
        if self._command_mode:
            self._task.reset(self._robot, self._command_mode)
        else:
            self._task.reset(self._robot)
        self._velocity_estimator = self._vel_estimator_class(self._robot)

        # Enables rendering after the loading
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 1)

        friction_coefficient = np.random.normal(0.7, 0.2)
        self._robot.set_foot_friction(friction_coefficient)

        return self.get_state(terrain_enc=terrain_enc)

    def _wait_for_rendering(self):
        """Sleep, otherwise the computation takes less time than real time
        To be finished. This is only used for real time rendering on GUI if the
        simulation is going too fast.
        """
        pass

    def get_action_dims(self):
        """Number of action dimensions outputted by the policy"""
        return self._pmtg.get_num_actions()

    def get_observation_dims(self, use_terrain_enc=False):
        """Number of observation dimensions used as input to the policy"""
        if use_terrain_enc:
            return self._pmtg.get_num_states() + self._robot_observation_dims + self._velocity_dims*2 + 6
        else:
            return self._pmtg.get_num_states() + self._robot_observation_dims + self._velocity_dims*2

    def hard_reset(self, heightfield=None, terrain_enc=None, is_slope=False):
        """Destroys the simulation and resets everything, including the scene,
        the robot and all the objects. All the instances are destroyed and new
        ones are created from the classes."""

        self._env_step_counter = 0
        self._sim_step_counter = 0
        return self._load(heightfield, terrain_enc=terrain_enc, is_slope=is_slope)

    def soft_reset(self, terrain_enc=None):
        """Resets the robot pose, position and orientation in the simulation. 
        This might be faster than a hard reset, however if there dynamical 
        objects whose positions might have changed, it is better to hard_reset

        Args:
            terrain_enc: Provides vector encoding of terrain as part of state information for training on
             varied terrains.
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
        if self._command_mode:
            self._task.reset(self._robot, self._command_mode)
        else:
            self._task.reset(self._robot)
        self._velocity_estimator.reset()

        return self.get_state(terrain_enc)

    def get_foot_position_in_hip_frame(self):
        """Returns x,y,z position of every foot in hip frame"""
        return self._robot.GetFootPositionsInBaseFrame()

    def get_foot_desired_position_in_hip_frame(self):
        """Returns desired x,y,z position of every foot in hip frame"""
        return self._pmtg.get_desired_foot_position()

    def get_state(self, terrain_enc=None):
        """ Returns observation only - used when using the reset methods.
        If a terrain encoding is provided, this is communicated as part of the state information."""
        # STATE WITH YAW RATE
        if terrain_enc is not None:
            state = np.concatenate((self._robot.get_observation(),
                                    self._pmtg.get_phase(),
                                    np.array([self._velocity_estimator.get_x_y_velocities()]).reshape(-1),
                                    np.array([self._velocity_estimator.get_measurements()[-1]]).reshape(-1),
                                    np.array([self._task.get_command()]).reshape(-1),
                                    np.array(terrain_enc).reshape(-1)), axis=0)
        else:
            state = np.concatenate((self._robot.get_observation(),
                                    self._pmtg.get_phase(),
                                    np.array([self._velocity_estimator.get_x_y_velocities()]).reshape(-1),
                                    np.array([self._velocity_estimator.get_measurements()[-1]]).reshape(-1),
                                    np.array([self._task.get_command()]).reshape(-1)), axis=0)
        return state

    def get_state_reward_done(self, action, index=None, terrain_enc=None):
        """Returns the state, the reward and the boolean for the current time step.
        If a terrain encoding is provided, this is communicated as part of the state information.
        Index is used to compute TG switching reward term. """
        # STATE WITH YAW RATE
        if terrain_enc is not None:
            state = np.concatenate((self._robot.get_observation(),
                                    self._pmtg.get_phase(),
                                    np.array([self._velocity_estimator.get_x_y_velocities()]).reshape(-1),
                                    np.array([self._velocity_estimator.get_roll_pitch_yaw_rate()[2]]).reshape(-1),
                                    np.array([self._task.get_command()]).reshape(-1),
                                    np.array(terrain_enc).reshape(-1)), axis=0)
        else:
            state = np.concatenate((self._robot.get_observation(),
                                    self._pmtg.get_phase(),
                                    np.array([self._velocity_estimator.get_x_y_velocities()]).reshape(-1),
                                    np.array([self._velocity_estimator.get_roll_pitch_yaw_rate()[2]]).reshape(-1),
                                    np.array([self._task.get_command()]).reshape(-1)), axis=0)

        reward, r_lv, r_av, r_s, r_br, r_bp, r_t = self._task.get_reward(self._velocity_estimator.get_measurements(),
                                                                         action, index)
        done = self._task.check_default_terminal_condition()
        info = np.array([r_lv, r_av, r_s, r_br, r_bp, r_t])

        return state, reward, done, info

    def increase_speed(self, delta):
        """Increases the speed of the TG by a delta factor. Only used when
        testing the trajectory generator alone"""
        self._pmtg.increase_speed(delta)

    def focus_camera_on_robot(self):
        """Sets the camera to follow the robot. Has to be called at every time
        step. Good for taking videos on GUI connection"""
        self._pybullet_client.resetDebugVisualizerCamera(1.0, -60, -30,
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

    def step(self, action=None, terrain_enc=None):
        """Step forward the simulation, given an action of the robot.

        Args:
            action: The x,y,z coordinates of each foot link.
            terrain_enc: Encoding of the current terrain used to pass as part of state information.
            
        Returns:
            state: a dictionary where the keys are
                the sensor names and the values are the sensor readings.
            reward: the reward obtained by the robot. For the moment it will be
                Null since we are not doing RL but just testing out the env
            done: whether or not the episode terminated.
        """

        self._sim_step_counter += self._sim_time_step * self._num_action_repeat
        self._env_step_counter += self._num_action_repeat

        link_positions, index = self._pmtg.step(time=self._sim_step_counter, action=action)

        for _ in range(self._num_action_repeat):
            self._robot.inverse_kinematics_action(link_positions)
        self._velocity_estimator.update()

        link_positions = link_positions + a1.FOOT_POSITION_IN_HIP_FRAME

        # INCREASE FORWARD VELOCITY TO 0.6ms-1, STAY THERE AND DECREASE TO 0ms-1
        if self._env_step_counter <= 300 * self._num_action_repeat:
            self._task.change_speed_linearly("x", 1)
        elif 700*self._num_action_repeat < self._env_step_counter:
            self._task.change_speed_linearly("x", -1)

        # TEST 1
        # if self._env_step_counter <= 200 * self._num_action_repeat:
        #     self._task.change_speed_linearly("x", 1)
        # elif 700 * self._num_action_repeat < self._env_step_counter <= 750 * self._num_action_repeat:
        #     self._task.change_speed_linearly("x", 1)
        # elif 1700 * self._num_action_repeat < self._env_step_counter:
        #     self._task.change_speed_linearly("x", -1)

        # TEST 2
        # if self._env_step_counter <= 800*self._num_action_repeat:
        #     self._task.change_speed_linearly("x", 1)
        # elif 1200 * self._num_action_repeat < self._env_step_counter <= 1500*self._num_action_repeat:
        #     self._task.change_speed_linearly("x", -1)
        # elif 2000 * self._num_action_repeat < self._env_step_counter:
        #     self._task.change_speed_linearly("x", -1)

        # CONSTANT VEL INCREASE
        # self._task.change_speed_linearly("x", 1)

        return self.get_state_reward_done(link_positions[:12], index, terrain_enc)
