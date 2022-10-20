"""Adapted from Davide Paglieri repository available at: https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2020-2021/davide_paglieri/control_pmtg
Loads a PyBullet environment for the A1 robot locomotion"""

import pybullet as p
import pybullet_utils.bullet_client as bc

from environment import locomotion_env, simulation_env
from robot import a1
from pmtg import pmtg_wrapper
from tasks import control_velocity, command_conditioned_control


def load(connection_mode="DIRECT",
         task_name="control_velocity",
         environment="locomotion",
         command_mode=None,
         archive_filename='',
         tg_select=False,
         default_params=None):
    """Create an instance of the locomotion environment in PyBullet

    Args:
        connection_mode: The type of connection with the PyBullet client.
            Can be DIRECT or GUI. Defaults to DIRECT
        environment: The environment used for the simulation.
            Can be 'locomotion' for training an agent or 'tg_simulation' for generating the MAP-Elites archive.
        task_name: The task assigned to the robot.
            It can be "control_velocity" where commands are defined by the step method of the locomotion environment or
            "command_conditioned" where commands are sampled according to a random goal position.
        command_mode: If using the "command_conditioned" task, specify the command mode as one of: "fixed_dir",
            "straight_x", "straight_y" or "random".
        archive_filename: Name of file containing collection of TGs.
        tg_select: If True, the policy can select a TG, otherwise it uses the same one that is instantiated throughout.
        default_params: Used to simulate TG parameters for MAP-Elites archive generation

    Returns:
        env: the locomotion environment
    """

    pmtg = pmtg_wrapper.PMTG(archive_filename=archive_filename, tg_select=tg_select)

    if connection_mode == "DIRECT":
        client = bc.BulletClient(p.DIRECT)
    elif connection_mode == "GUI":
        client = bc.BulletClient(p.GUI)
    else:
        raise ("Connection mode not supported")

    if task_name == "control_velocity":
        task = control_velocity.Task(pmtg)
    elif task_name == "command_conditioned":
        task = command_conditioned_control.Task(pmtg)

    client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    if environment == "locomotion":
        return locomotion_env.LocomotionEnv(pybullet_client=client,
                                            robot_class=a1.A1,
                                            urdf_path=a1.URDF_PATH,
                                            pmtg=pmtg,
                                            task=task,
                                            command_mode=command_mode,
                                            scene_class=None)
    elif environment == "tg_simulation":
        return simulation_env.SimulationEnv(pybullet_client=client,
                                            robot_class=a1.A1,
                                            urdf_path=a1.URDF_PATH,
                                            pmtg=pmtg_wrapper.PMTG(default_tg_params=default_params),
                                            scene_class=None)
