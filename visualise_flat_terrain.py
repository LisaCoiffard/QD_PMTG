import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from environment import env_loader
from train_flat_terrain import Policy
from scenes.terrain_generator import create_slope_terrain
from robot import a1

sns.set_theme(style='darkgrid', palette='colorblind')


def run_policy(filename, episode_length, archive_filename=None, tg_select=None, is_slope=False):
    speed_1_list = []
    speed_2_list = []
    speed_3_list = []
    speed_4_list = []

    robot_x_vels = []
    robot_y_vels = []
    desired_x_vels = []
    desired_y_vels = []
    robot_yaw_rate = []
    desired_yaw_rate = []

    front_x = []
    front_y = []
    front_z = []
    rear_x = []
    rear_y = []
    rear_z = []

    env = env_loader.load('GUI',
                          task_name='control_velocity',
                          archive_filename=archive_filename,
                          tg_select=tg_select)
    nb_inputs = env.get_observation_dims()
    nb_outputs = env.get_action_dims()
    policy = Policy(nb_inputs, nb_outputs, None)
    policy.load_policy(filename)

    if is_slope:
        obs = env.hard_reset(heightfield=create_slope_terrain(10), is_slope=is_slope)
    else:
        obs = env.soft_reset()

    done = False
    total_reward = 0
    total_lv_reward = 0
    total_steps = 0

    # env.start_recording()

    while not done and total_steps < episode_length:

        env.focus_camera_on_robot()

        policy.observe(obs)
        obs = policy.normalize(obs)
        action = policy.evaluate(obs, None, None, None)
        obs, reward, done, info = env.step(action)

        speed_1_list.append(action[12] * 1.25 + 1.25)
        speed_2_list.append(action[13] * 1.25 + 1.25)
        speed_3_list.append(action[14] * 1.25 + 1.25)
        speed_4_list.append(action[15] * 1.25 + 1.25)

        robot_x_vels.append(obs[-6])
        robot_y_vels.append(obs[-5])
        robot_yaw_rate.append(obs[-4])
        desired_x_vels.append(obs[-3])
        desired_y_vels.append(obs[-2])
        desired_yaw_rate.append(obs[-1])

        if (750 < total_steps < 1000):
            front_x.append(action[0] * 0.1)
            front_y.append(action[1] * 0.1)
            front_z.append(action[2] * 0.05)
            rear_x.append(action[9] * 0.1)
            rear_y.append(action[10] * 0.1)
            rear_z.append(action[11] * 0.05)

        total_reward += reward
        total_lv_reward += info[0]
        total_steps += 1

    # env.stop_recording()

    print(f"Reward: {total_reward}")
    print(total_steps)

    velocities_array = np.vstack([desired_x_vels, desired_y_vels, robot_x_vels, robot_y_vels])
    yaw_rates_array = np.vstack([desired_yaw_rate, robot_yaw_rate])

    speed_1_list = gaussian_filter1d(speed_1_list, sigma=1)
    speed_2_list = gaussian_filter1d(speed_2_list, sigma=1)
    speed_3_list = gaussian_filter1d(speed_3_list, sigma=1)
    speed_4_list = gaussian_filter1d(speed_4_list, sigma=1)

    tg_params_array = np.vstack([speed_1_list, speed_2_list, speed_3_list, speed_4_list])

    front_residuals_array = np.vstack([front_x, front_y, front_z])
    rear_residuals_array = np.vstack([rear_x, rear_y, rear_z])

    return velocities_array, yaw_rates_array, tg_params_array, front_residuals_array, rear_residuals_array


def plot_residuals(front_res, rear_res):
    front_x = front_res[0]
    front_y = front_res[1]
    front_z = front_res[2]

    rear_x = rear_res[0]
    rear_y = rear_res[1]
    rear_z = rear_res[2]

    fig, axs = plt.subplots(2)

    axs[0].set_title("Front Right Residuals")
    axs[0].plot(front_x, label=r"$x$ residuals", color='b')
    axs[0].plot(front_y, label=r"$y$ residuals", color='g')
    axs[0].plot(front_z, label=r"$z$ residuals", color='r')
    axs[0].legend()
    axs[0].set_ylabel("Residual Values")

    axs[1].set_title("Rear Left Residuals")
    axs[1].plot(rear_x, label=r"$x$ residuals", color='b')
    axs[1].plot(rear_y, label=r"$y$ residuals", color='g')
    axs[1].plot(rear_z, label=r"$z"
                              r"$ residuals", color='r')
    axs[1].legend()
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("Residual Values")

    fig.tight_layout(pad=1)
    fig.set_figheight(4)
    fig.set_figwidth(12)
    fig.savefig("Residuals", dpi=fig.dpi)


def plot_parameters(tg_params):
    speed_1_list = tg_params[0]
    speed_2_list = tg_params[1]
    speed_3_list = tg_params[2]
    speed_4_list = tg_params[3]

    fig, axs = plt.subplots(1, figsize=(10, 6))

    axs.set_title("TG parameters", fontsize=18)
    axs.plot(speed_1_list, label="Frequency Multiplier leg 1", color='b')
    axs.plot(speed_2_list, label="Frequency Multiplier leg 2", color='c')
    axs.plot(speed_3_list, label="Frequency Multiplier leg 3", color='red')
    axs.plot(speed_4_list, label="Frequency Multiplier leg 4", color='m')

    axs.legend(fontsize=18)
    axs.set_xlabel("Time step", fontsize=18)
    axs.set_ylabel("Parameter Values", fontsize=18)

    fig.tight_layout(pad=0.0)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    fig.savefig("TG parameters", dpi=fig.dpi)


def plot_speed_profile(vels, yaw_rates):
    desired_x_vels = vels[0]
    robot_x_vels = vels[2]
    desired_y_vels = vels[1]
    robot_y_vels = vels[3]

    desired_yaw_rate = yaw_rates[0]
    robot_yaw_rate = yaw_rates[1]

    fig, axs = plt.subplots(3, figsize=(10, 6))

    axs[0].set_title("Speed profile")
    axs[0].plot(desired_x_vels, label=r"Target $x$-velocity", color='b', ls='--')
    axs[0].plot(robot_x_vels, label=r"Robot $x$-velocity", color='g', ls='-')
    axs[0].legend()
    axs[0].set_ylabel(r"$x$-velocity (m/s)")

    axs[1].plot(desired_y_vels, label=r"Target $y$-velocity", color='b', ls='--')
    axs[1].plot(robot_y_vels, label=r"Robot $y$-velocity", color='g', ls='-')
    axs[1].legend()
    axs[1].set_ylabel(r"$y$-velocity (m/s)")

    axs[2].plot(desired_yaw_rate, label="Target yaw rate", color='m', ls='--')
    axs[2].plot(robot_yaw_rate, label="Robot yaw rate", color='c', ls='-')
    axs[2].legend()
    axs[2].set_xlabel("Time step")
    axs[2].set_ylabel("Yaw rate (rad/s)")

    fig.tight_layout(pad=0.0)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    fig.savefig("Velocity and yaw rate profile", dpi=fig.dpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--filename', help='Directory root where results are stored', type=str, default='results')
    parser.add_argument(
        '--ep_length', help='Total number of steps per episode', type=int, default=1000)
    parser.add_argument(
        '--command_mode', help='Command mode for the task', type=str, default='fixed_dir')
    parser.add_argument('--tg_select', help='Enable TG selection by the policy', type=int, default=0)
    parser.add_argument('--archive', help='Archive file name', type=str, default='')
    parser.add_argument('--slope', help="Test the trained controller on a sloped terrain", type=int, default=0)

    args = parser.parse_args()

    velocities_array, yaw_rates_array, \
    tg_params_array, front_residuals_array, \
    rear_residuals_array = run_policy(args.filename, args.ep_length, args.archive, args.tg_select, args.slope)
