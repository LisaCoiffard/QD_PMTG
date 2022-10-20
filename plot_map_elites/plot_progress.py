#! /usr/bin/env python
# | This file is a part of the pymap_elites framework.
# | Copyright 2019, INRIA
# | Main contributor(s):
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# |
# |
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
# |
# | This software is governed by the CeCILL license under French law
# | and abiding by the rules of distribution of free software.  You
# | can use, modify and/ or redistribute the software under the terms
# | of the CeCILL license as circulated by CEA, CNRS and INRIA at the
# | following URL "http://www.cecill.info".
# |
# | As a counterpart to the access to the source code and rights to
# | copy, modify and redistribute granted by the license, users are
# | provided only with a limited warranty and the software's author,
# | the holder of the economic rights, and the successive licensors
# | have only limited liability.
# |
# | In this respect, the user's attention is drawn to the risks
# | associated with loading, using, modifying and/or developing or
# | reproducing the software by the user in light of its specific
# | status of free software, that may mean that it is complicated to
# | manipulate, and that also therefore means that it is reserved for
# | developers and experienced professionals having in-depth computer
# | knowledge. Users are therefore encouraged to load and test the
# | software's suitability as regards their requirements in conditions
# | enabling the security of their systems and/or data to be ensured
# | and, more generally, to use and operate it in the same conditions
# | as regards security.
# |
# | The fact that you are presently reading this means that you have
# | had knowledge of the CeCILL license and that you accept its terms.
import glob
from pylab import *
import brewer2mpl
import numpy as np
import sys
import seaborn as sns
import math
import gzip
import matplotlib.gridspec as gridspec

from collections import defaultdict
from matplotlib import pyplot as plt

sns.set_theme(style='darkgrid', palette='colorblind')


params = {
    'axes.labelsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [12, 4]
}
rcParams.update(params)


def customize_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis='y', length=0)
    # ax.get_yaxis().tick_left()

    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(('outward', 5))
    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='--', linewidth=1)


fig = figure(frameon=False)  # no frame

# plt.box(False)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))


ax1 = fig.add_subplot(131)
labels = ['Version 1', 'Version 2']

for i in range(len(sys.argv[1:-1])):
    data = np.loadtxt(sys.argv[i+1])
    label = labels[i]
    ax1.plot(data[:, 0], data[:, 1], '-', linewidth=2, label=label)
    ax1.set_title('Coverage')

customize_axis(ax1)

ax2 = fig.add_subplot(132)
for i in range(len(sys.argv[1:-1])):
    data = np.loadtxt(sys.argv[i+1])
    label = labels[i]
    ax2.plot(data[:, 0], data[:, 3], '-', linewidth=2, label=label)
    ax2.set_title('Mean fitness')

customize_axis(ax2)

ax3 = fig.add_subplot(133)
ax3.grid(axis='y', color="0.9", linestyle='--', linewidth=1)
for i in range(len(sys.argv[1:-1])):
    data = np.loadtxt(sys.argv[i+1])
    label = labels[i]
    ax3.plot(data[:, 0], data[:, 2], '-', linewidth=2, label=label)
    ax3.set_title('Max fitness')

customize_axis(ax3)

legend = ax2.legend(loc=4)  # bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(3))
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('1.0')

fig.tight_layout()
plt.show()
fig.savefig(sys.argv[-1] + '/progress.png')