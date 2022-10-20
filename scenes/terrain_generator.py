"""Adapted from Davide Paglieri repository available at: https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2020-2021/davide_paglieri/control_pmtg"""

import numpy as np
from opensimplex import OpenSimplex
import math
import time
import random
import pybullet as p
from environment import env_loader

size = 256


def create_steps_map(amplitude, step):
    step = math.ceil(step) + 1
    a_ = np.zeros((size, size))
    a = np.random.random([size // step, size // step]) * amplitude
    a_[:(size // step * step), :(size // step) * step] = a.repeat(step, axis=0).repeat(step, axis=1)
    return a_


def create_opensimplex_map(amplitude, feature_size=24):
    num = random.randint(0, 1000)
    a = np.zeros((size, size))
    simplex = OpenSimplex()
    for x in range(size):
        for y in range(0, size):
            a[x, y] = simplex.noise2d((x + num) / feature_size, (y + num) / feature_size) * amplitude
    return a


def create_stairs_map(height, width):
    a = np.zeros((1, size))
    width = math.ceil(width)
    inc = 0
    i = 0
    while i < size:
        for j in range(i, i + 1 + width, 1):
            if j >= size:
                break
            a[0, j] = inc
        i += 1 + width
        # if not 46<i<64:
        inc += height
    return a.repeat(size, axis=0)


def create_slope_terrain(angle):
    a = np.zeros((1, size))
    slope = np.tan(np.radians(angle))
    slope *= 0.025

    a[0][int(size / 5):] = np.arange(0, size - int(size / 5))
    a[0][int(size / 5):] = slope * a[0][int(size / 5):]

    return a.repeat(size, axis=0).reshape(-1)


def create_terrain(feature_list):
    enc = feature_list * [0.1, 30, 0.1, 10, 0.05, 30]
    enc[3] += 10
    enc[5] += 8
    b = create_steps_map(enc[0], enc[1])
    c = create_opensimplex_map(enc[2], enc[3])
    d = create_stairs_map(enc[4], enc[5])
    return (b + c + d).reshape(-1)