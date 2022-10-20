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


# Setting the Hyper Parameters
class Hp:

    def __init__(self):
        self.nb_steps = 3000
        self.episode_length = 1500
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

        if message == _EXPLORE:
            policy = payload[0]
            hp = payload[1]
            direction = payload[2]
            delta = payload[3]
            state = env.soft_reset()
            done = False
            num_plays = 0.
            sum_rewards = 0
            while not done and num_plays < hp.episode_length:
                policy.observe(state)
                state = policy.normalize(state)
                action = policy.evaluate(state, delta, direction, hp)
                state, reward, done, _ = env.step(action)
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
def explore(env, policy, direction, delta, hp):
    state = env.soft_reset()
    # print(state)
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
        # print(action.shape)
        state, reward, done, info = env.step(action)
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

def train(env, policy, hp, parentPipes):
    reward_list = []
    r_lv_list = []
    r_av_list = []
    r_s_list = []
    r_br_list = []
    r_bp_list = []
    r_t_list = []
    num_plays_per_rollout = []
    num_steps_save = 50

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
                parentPipe.send([_EXPLORE, [policy, hp, "positive", deltas[k]]])
            for k in range(hp.nb_directions):
                positive_rewards[k] = parentPipes[k].recv()[0]

            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [policy, hp, "negative", deltas[k]]])
            for k in range(hp.nb_directions):
                negative_rewards[k] = parentPipes[k].recv()[0]

        # Otherwise use this for no multiprocessing - used for testing
        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(env, policy, "positive", deltas[k], hp)[0]

            # Getting the negative rewards in the negative directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(env, policy, "negative", deltas[k], hp)[0]

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

        # Printing the final reward of the policy after the update
        reward_evaluation, time_steps, r_lv, r_av, r_s, r_br, r_bp, r_t = explore(env, policy, None, None, hp)
        t2 = time.time()
        print("Time taken:", t2-t1)
        print('Step:', step, 'Reward:', reward_evaluation)
        print('Linear velocity reward:', r_lv)
        print('Angular velocity reward: ', r_av)
        print('Smoothness reward: ', r_s)
        print('Pitch/roll rewards: ', r_bp, r_br)
        print('Torque reward: ', r_t)
        print('Total steps: ', time_steps)

        if (step > hp.nb_steps - num_steps_save):
            policy.save("/seed_" + str(hp.seed) + "_" + str(step) + "_" + str(int(round(reward_evaluation))))

        num_plays_per_rollout.append(time_steps)
        reward_list.append(int(round(reward_evaluation)))
        r_lv_list.append(int(round(r_lv)))
        r_av_list.append(int(round(r_av)))
        r_s_list.append(int(round(r_s)))
        r_br_list.append(int(round(r_br)))
        r_bp_list.append(int(round(r_bp)))
        r_t_list.append(int(round(r_t)))

    # Save arrays with rewards and total number of steps per rollout
    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list.txt", "wb") as fp:
        pickle.dump(reward_list, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_num_plays_per_rollout.txt", "wb") as fp:
        pickle.dump(num_plays_per_rollout, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list_lv.txt", "wb") as fp:
        pickle.dump(r_lv_list, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list_av.txt", "wb") as fp:
        pickle.dump(r_av_list, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list_s.txt", "wb") as fp:
        pickle.dump(r_s_list, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list_br.txt", "wb") as fp:
        pickle.dump(r_br_list, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list_bp.txt", "wb") as fp:
        pickle.dump(r_bp_list, fp)

    with open(args.logdir + "/seed_" + str(hp.seed) + "_reward_list_t.txt", "wb") as fp:
        pickle.dump(r_t_list, fp)


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
    parser.add_argument('--replications', help='Number of replications', type=int, default=3)
    parser.add_argument('--steps', help='Number of steps', type=int, default=1000)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument(
        '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)
    parser.add_argument('--archive', help='Archive file name', type=str, default='')
    parser.add_argument('--tg_select', help='Enable TG selection by the policy', type=int, default=0)

    args = parser.parse_args()

    hp = Hp()

    for rep in range(args.replications):

        hp.seed = int(time.time()*1000)%10000 #randomise the seed
        hp.nb_steps = args.steps
        print("seed = ", hp.seed)
        np.random.seed(hp.seed)

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

        nb_inputs = env.get_observation_dims()
        nb_outputs = env.get_action_dims()
        policy = Policy(nb_inputs, nb_outputs, args)

        # FOR TRAINING ON SLOPE

        print("start training")
        print(args)
        train(env, policy, hp, parentPipes)

        if args.mp:
            for parentPipe in parentPipes:
                parentPipe.send([_CLOSE, "pay2"])

            for p in processes:
                p.join()
