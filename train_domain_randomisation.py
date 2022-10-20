# AI 2018
# Implementation of ARS adapted from PyBullet
# https://github.com/bulletphysics/bullet3

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# Importing the libraries
import pickle
import os
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Pipe
import argparse

from environment import env_loader
from scenes.terrain_generator import create_terrain

# Setting the Hyper Parameters
class Hp():

    def __init__(self):
        self.nb_steps = 5
        self.episode_length = 1000
        self.learning_rate = 0.01
        self.nb_directions = 48
        self.nb_best_directions = 24
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.01
        self.seed = 187


# Multiprocess Exploring the policy on one specific direction and over one episode
_RESET = 1
_CLOSE = 2
_EXPLORE = 3
_CHANGE = 4


def ExploreWorker(rank, childPipe, archive_filename='', tg_select=None):
    env = env_loader.load('DIRECT', 'control_velocity', archive_filename=archive_filename, tg_select=tg_select)

    n = 0
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not childPipe.poll(0.001):
                continue
            message, payload = childPipe.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if message == _RESET:
            observation_n = env.soft_reset()
            childPipe.send(["reset ok"])
            continue

        if message == _CHANGE:
            E = payload[0]
            env.hard_reset(E)
            childPipe.send(["change ok"])

        if message == _EXPLORE:
            policy = payload[0]
            hp = payload[1]
            direction = payload[2]
            delta = payload[3]
            terrain_enc = payload[4]
            state = env.soft_reset(terrain_enc)
            done = False
            num_plays = 0.
            sum_rewards = 0
            while not done and num_plays < hp.episode_length:
                policy.observe(state)
                state = policy.normalize(state)
                action = policy.evaluate(state, delta, direction, hp)
                state, reward, done, _ = env.step(action, terrain_enc=terrain_enc)
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards])
            continue

        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()


# Normalizing the states
class Normalizer:

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

    def set_n_mu_diff(self, n, mu, diff):
        self.n = n
        self.mean = mu
        self.mean_diff = diff

    def save_n_mu_diff(self):
        self._n = self.n
        self._mean = self.mean
        self._mean_diff = self.mean_diff

    def get_saved_n_mu_diff(self):
        return self._n, self._mean, self._mean_diff


class Policy:

    def __init__(self, input_size, output_size, args):
        self.normalizer = Normalizer(input_size)
        try:
            policy = np.load(args.policy)
            self.theta = policy['arr_0']
            pre_trained_input = self.theta.shape[1]
            self.theta = np.concatenate((self.theta, np.zeros((output_size, input_size-pre_trained_input))), axis=1)
        except:
            self.theta = np.zeros((output_size, input_size))

        print("Starting policy theta=", self.theta)
        parameters = self.theta.size
        print(f"{parameters = }")

    def load_policy(self, filename):
        self.filename = filename
        _policy = np.load(filename)

        self.theta = _policy['arr_0']

        self.normalizer.set_n_mu_diff(_policy['arr_1'],
                                      _policy['arr_2'],
                                      _policy['arr_3'])

    def evaluate(self, input, delta, direction, hp):
        if direction is None:
            return np.tanh(self.theta.dot(input))
        elif direction == "positive":
            return np.tanh((self.theta + hp.noise * delta).dot(input))
        else:
            return np.tanh((self.theta - hp.noise * delta).dot(input))

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

    def save(self, name):
        print(f"Save {name}")
        self.normalizer.save_n_mu_diff()
        n, mu, diff = self.normalizer.get_saved_n_mu_diff()
        np.savez(args.logdir + name, self.theta, n, mu, diff, allow_pickle=True)

    def observe(self, x):
        self.normalizer.observe(x)

    def normalize(self, inputs):
        return self.normalizer.normalize(inputs)


# Exploring the policy on one specific direction and over one episode
def explore(env, heightfield, policy, direction, delta, hp, terrain_enc=None):
    # print("reset environment - includes resetting task, commands and goal")
    state = env.hard_reset(heightfield, terrain_enc=terrain_enc)
    done = False
    num_plays = 0.
    sum_rewards = 0
    sum_rewards_lv = 0
    sum_rewards_av = 0
    sum_rewards_s = 0
    sum_rewards_br = 0
    sum_rewards_bp = 0
    sum_rewards_t = 0
    while not done and num_plays < hp.episode_length:
        # print(num_plays)
        policy.observe(state)
        state = policy.normalize(state)
        action = policy.evaluate(state, delta, direction, hp)
        # print(action)
        state, reward, done, info = env.step(action, terrain_enc=terrain_enc)
        # print("COMMAND:", state[-3:])
        # print(info)
        sum_rewards_lv += info[0]
        sum_rewards_av += info[1]
        sum_rewards_s += info[2]
        sum_rewards_br += info[3]
        sum_rewards_bp += info[4]
        sum_rewards_t += info[5]
        # reward += reward#max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1

    # print(sum_rewards_lv, sum_rewards_av, sum_rewards_s, sum_rewards_br, sum_rewards_bp, sum_rewards_t)
    return sum_rewards, num_plays, sum_rewards_lv, sum_rewards_av, sum_rewards_s, sum_rewards_br, sum_rewards_bp, sum_rewards_t


# Training the AI

def train(env, heightfield, policy, hp, parentPipes, terrain_enc=None):

    if parentPipes:
        for k in range(hp.nb_directions):
            parentPipe = parentPipes[k]
            parentPipe.send([_CHANGE, [heightfield]])
        for k in range(hp.nb_directions):
            parentPipes[k].recv()

    for step in range(hp.nb_steps):

        t1 = time.time()

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        # If multiprocessing is enables
        if parentPipes:
            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [policy, hp, "positive", deltas[k], terrain_enc]])
            for k in range(hp.nb_directions):
                positive_rewards[k] = parentPipes[k].recv()[0]

            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [policy, hp, "negative", deltas[k], terrain_enc]])
            for k in range(hp.nb_directions):
                negative_rewards[k] = parentPipes[k].recv()[0]

        # Otherwise use this for no multiprocessing - used for testing
        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(env, heightfield, policy, "positive", deltas[k], hp, terrain_enc=terrain_enc)[0]

            # Getting the negative rewards in the negative directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(env, heightfield, policy, "negative", deltas[k], hp, terrain_enc=terrain_enc)[0]

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        t2 = time.time()
        print("Time taken:", t2-t1)

    return policy


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Running the main code
if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env', help='Gym environment name', type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--steps', help='Number of steps', type=int, default=1)
    parser.add_argument('--iterations', help='Number iterations', type=int, default=500)
    parser.add_argument('--replications', help='Number of random seed replications', type=int, default=3)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument(
        '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)
    parser.add_argument('--archive', help='Archive file name', type=str, default='')
    parser.add_argument('--tg_select', help='Enable TG selection by the policy', type=int, default=0)

    args = parser.parse_args()

    hp = Hp()
    hp.nb_steps = args.steps

    for rep in range(args.replications):

        print("REP:", rep)

        # Initialise separate threads for each direction
        parentPipes = None
        if args.mp:
            num_processes = hp.nb_directions
            processes = []
            childPipes = []
            parentPipes = []

            for pr in range(num_processes):
                parentPipe, childPipe = Pipe()
                parentPipes.append(parentPipe)
                childPipes.append(childPipe)

            for rank in range(num_processes):
                p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank], args.archive, args.tg_select))
                p.start()
                processes.append(p)

        env = env_loader.load('DIRECT', 'control_velocity', archive_filename=args.archive, tg_select=args.tg_select)

        nb_inputs = env.get_observation_dims(use_terrain_enc=True)
        nb_outputs = env.get_action_dims()
        policy = Policy(nb_inputs, nb_outputs, args)

        E_list = np.array([
            [0, 0, 0, 0, 0, 0],

            [0.20, 0.90, 0, 0, 0, 0],
            [0.40, 0.70, 0, 0, 0, 0],
            [0.60, 0.50, 0, 0, 0, 0],
            [0.70, 0.40, 0, 0, 0, 0],
            [0.80, 0.30, 0, 0, 0, 0],
            [0.90, 0.20, 0, 0, 0, 0],

            [0, 0, 0.20, 0.10, 0, 0],
            [0, 0, 0.35, 0.30, 0, 0],
            [0, 0, 0.45, 0.40, 0, 0],
            [0, 0, 0.55, 0.50, 0, 0],
            [0, 0, 0.65, 0.60, 0, 0],
            [0, 0, 0.75, 0.70, 0, 0],

            [0, 0, 0, 0, 0.30, 0.80],
            [0, 0, 0, 0, 0.50, 0.60],
            [0, 0, 0, 0, 0.60, 0.50],
            [0, 0, 0, 0, 0.70, 0.40],
            [0, 0, 0, 0, 0.80, 0.30],
            [0, 0, 0, 0, 0.90, 0.20],
        ])

        print("start training")
        print(args)

        reward_list = []
        r_lv_list = []
        r_av_list = []
        r_s_list = []
        r_br_list = []
        r_bp_list = []
        r_t_list = []
        num_plays_per_rollout = []
        num_it_save = 50

        for it in range(args.iterations):

            hp.seed = int(time.time() * 1000) % 10000  # randomise the seed
            print("seed = ", hp.seed)
            np.random.seed(hp.seed)

            terrain_idx = np.random.randint(0, len(E_list))
            terrain_enc = E_list[terrain_idx]
            # print(terrain_enc)
            policy = train(env, create_terrain(terrain_enc), policy, hp, parentPipes, terrain_enc)

            total_reward = 0
            for enc in E_list:
                heightfield = create_terrain(enc)
                evaluation_rewards = explore(env, heightfield, policy, None, None, hp, terrain_enc=enc)
                total_reward += evaluation_rewards[0]

            print('Step:', it, 'Reward:', total_reward)

            if (it > args.iterations - num_it_save):
                policy.save("/rep_" + str(rep) + "_" + str(it) + "_" + str(int(round(total_reward))))

            reward_list.append(int(round(total_reward)))

            # Save arrays with rewards and total number of steps per rollout
        with open(args.logdir + "/rep_" + str(rep) + "_reward_list.txt", "wb") as fp:
            pickle.dump(reward_list, fp)

        if args.mp:
            for parentPipe in parentPipes:
                parentPipe.send([_CLOSE, "pay2"])

            for p in processes:
                p.join()
