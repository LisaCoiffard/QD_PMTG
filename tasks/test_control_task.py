import unittest
import pybullet as p
import pybullet_utils.bullet_client as bc
import numpy as np

from pmtg import pmtg_wrapper
from tasks.command_conditioned_control import Task, Command, Goal
from robot import a1


class TestTask(unittest.TestCase):

    def setUp(self):
        print("setting up")
        client = bc.BulletClient(p.DIRECT)
        client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        dt = 0.0005
        urdf_path = a1.URDF_PATH
        quadruped = client.loadURDF(urdf_path, a1.INIT_POSITION, a1._IDENTITY_ORIENTATION)

        self.robot = a1.A1(client, quadruped, dt)
        self.pmtg = pmtg_wrapper.PMTG()

    def tearDown(self):
        print("tearing down")

    def test_task_init(self):
        task = Task(self.pmtg, self.robot, command_mode="straight")

        # test resetting the command
        self.assertEqual(task.command.command[0], 0)
        self.assertEqual(task.command.command[1], 0)
        self.assertEqual(task.command.command[2], 0)
        # test resetting the command direction
        self.assertEqual(task.command.command_dir, 0)
        # test the command mode is properly set
        self.assertEqual(task.command.command_mode, "straight")
        # test resetting the goal position
        self.assertEqual(task.command._goal.goal_pos[0], 0)
        self.assertEqual(task.command._goal.goal_pos[1], 0)

    def test_reset_task(self):

        task = Task(self.pmtg, self.robot, command_mode="random")
        task.reset(self.robot)
        print("Sampled command at reset:", task.command.command)
        print("Updated command direction:", task.command.command_dir)
        print("Sampled goal position at reset:", task.command._goal.goal_pos)


class TestCommand(unittest.TestCase):

    def setUp(self):
        print("setting up")
        self.client = bc.BulletClient(p.DIRECT)
        self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        dt = 0.0005
        urdf_path = a1.URDF_PATH
        self.quadruped = self.client.loadURDF(urdf_path, a1.INIT_POSITION, a1._IDENTITY_ORIENTATION)

        self.robot = a1.A1(self.client, self.quadruped, dt)
        self.pmtg = pmtg_wrapper.PMTG()

    def tearDown(self):
        print("tearing down")

    def test_update_command(self):

        # test update command for fixed_dir mode
        command = Command(command_mode="fixed_dir", robot=self.robot)
        command.sample_command()
        command.update_command()
        command_prev = command.command
        command.update_command()
        command_new = command.command
        np.testing.assert_array_equal(command_prev, command_new)

        # test update command for straight mode
        command = Command(command_mode="straight", robot=self.robot)
        command.sample_command()
        command_prev = command.command
        command.update_command()
        command_new = command.command
        np.testing.assert_array_equal(command_prev, command_new)
        # update robot position to check if goal and command are updated
        robot_pos = np.concatenate((command._goal.goal_pos, [self.robot.get_base_position()[-1]]))
        self.client.resetBasePositionAndOrientation(self.quadruped, robot_pos, a1._IDENTITY_ORIENTATION)
        command_prev = command_new.copy()
        command.update_command()
        command_new = command.command
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(command_prev, command_new)
        self.assertEqual(command_prev[-1], command_new[-1])

        # test update command for random mode
        command = Command(command_mode="random", robot=self.robot)
        command.sample_command()
        command_prev = command.command
        command.update_command()
        command_new = command.command
        np.testing.assert_array_equal(command_prev, command_new)
        # update robot position to check if goal and command are updated
        robot_pos = np.concatenate((command._goal.goal_pos, [self.robot.get_base_position()[-1]]))
        self.client.resetBasePositionAndOrientation(self.quadruped, robot_pos, a1._IDENTITY_ORIENTATION)
        command_prev = command_new.copy()
        command.update_command()
        command_new = command.command
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(command_prev, command_new)
        self.assertNotEqual(command_prev[-1], command_new[-1])

    def test_sample_command(self):

        # test sample command for fixed_dir mode
        command = Command(command_mode="fixed_dir", robot=self.robot)
        command.sample_command()
        np.testing.assert_array_equal(command.command, np.array([1, 0, 0]))
        np.testing.assert_array_equal(command._goal.goal_pos, np.array([0, 0]))
        self.assertEqual(command.command_dir, 0)

        # test sample command for straight mode
        command = Command(command_mode="straight", robot=self.robot)
        command.sample_command()
        # commands should be roughly in the same direction (no large changes) with yaw rate of zero
        for i in range(10):
            command_dir_prev = command.command_dir
            command_prev = command.command
            command.sample_command()
            command_dir_new = command.command_dir
            command_new = command.command
            np.testing.assert_allclose(command_dir_prev, command_dir_new, rtol=0.5)
            self.assertEqual(command_prev[-1], command_new[-1])
            self.assertEqual(command_new[-1], 0)

        # test sample command for random mode
        command = Command(command_mode="random", robot=self.robot)
        command.sample_command()
        for i in range(10):
            command_dir_prev = command.command_dir
            command_prev = command.command
            command.sample_command()
            command_dir_new = command.command_dir
            command_new = command.command
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(command_prev, command_new)
                np.testing.assert_array_equal(command_dir_prev, command_dir_new)


class TestGoal(unittest.TestCase):

    def setUp(self):
        print("setting up")
        client = bc.BulletClient(p.DIRECT)
        client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        dt = 0.0005
        urdf_path = a1.URDF_PATH
        quadruped = client.loadURDF(urdf_path, a1.INIT_POSITION, a1._IDENTITY_ORIENTATION)

        self.robot = a1.A1(client, quadruped, dt)
        self.pmtg = pmtg_wrapper.PMTG()

    def tearDown(self):
        print("tearing down")

    def test_sample_goal(self):

        # test goal sampled for straight command
        goal = Goal(self.robot, command_mode="straight")
        goal.sample_goal()
        self.assertNotEqual(goal.goal_pos[0], self.robot.get_base_position()[0])
        self.assertEqual(goal.goal_pos[1], self.robot.get_base_position()[1] + goal.terrain_size_y*0.4)

        # test goal sampled for random command
        goal = Goal(self.robot, command_mode="random")
        goal.sample_goal()
        self.assertNotEqual(goal.goal_pos[0], self.robot.get_base_position()[0])
        self.assertNotEqual(goal.goal_pos[1], self.robot.get_base_position()[1])
