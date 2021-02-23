"""
Experiments for studying the learned mean embedding.

Two experiments:
    * PCA with several different dynamics.
    * Interpolating the latent space between two dynamics.
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
import time

import seaborn as sns
from matplotlib.pyplot import rc as rc

from tqdm import tqdm
from omegaconf import OmegaConf

from utils import context_target_split
from models import MLPModel

sns.set_style('darkgrid')
sns.set_style('white')
rc('font', family='serif')
rc('text', usetex=True)

# experiment loader



#########################
#Change these
dataname = 'sinefreq'
sigma = 0.1 # for noisysine
process_choice = 'nodep'
length_context = 15
start_point = 2
darkness =0.15
truth_darkness = 1
colour = '#9013FE'
number_to_plot = 35
#########################

model_name = 'full_T_20210222-115455'

# make the save folder
save_folder = '../figures/' + model_name + '-'
try:
        os.makedirs('./'+save_folder)
except FileExistsError:
        pass

conf = OmegaConf.load("../runs/" + dataname + '/' + model_name +'/conf.yaml')
h_sizes = OmegaConf.to_container(conf.hidden_sizes)
model = MLPModel(dim_y=1, dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                 hidden_sizes_encoder=h_sizes,
                 hidden_sizes_ode_net=h_sizes,
                 hidden_sizes_decoder=h_sizes,
                 t0=-3.2, batch_size=100)

model.load_state_dict(torch.load('../runs'  + '/' + dataname+ '/' +  model_name + '/seed_0/model.pth'))

results = []
num_context = 15
num_extra_target = 50
num_points = 100
locations = torch.Tensor(np.random.choice(num_points, size=num_context + num_extra_target, replace=False)).long()
freqs = [1.0, 1.5, 1.9]
for xp, freq in enumerate(freqs):
        #import data
        from datasets import FreqSineData
        from math import pi
        dataset = FreqSineData(amplitude_range=(0.5, 1.),
                                    shift_range=(-.0000, .00001),
                                    freq_range=(freq, freq + 1e-5),
                                    num_samples=100)
        x_min, x_max = -pi, pi
        y_min, y_max = -1.1, 1.1
        initial_x = -3.2
        t = torch.cat([torch.Tensor(t).unsqueeze(0) for t, x in dataset.data])
        x = torch.cat([torch.Tensor(x).unsqueeze(0) for t, x in dataset.data])

        # fix locations so all times series have the same context
        t_context, x_context, t_target, _ = context_target_split(t, x,
                                                                 num_context,
                                                                 num_extra_target,
                                                                 locations=locations)

        # compute q(z|C)
        p_y_pred, _, q_context = model(t_context, x_context, t_target, z=None)
        mu_context = q_context.loc
        sigma_context = q_context.scale
        results.append((freq, mu_context, sigma_context))


# Experiment 1: interpolation
# Decode from mu_2 to mu_1 along mu_1 - mu_2.

# select mu for dynamics (freq = 1)
mu_1 = results[0][1][0]
# select mu for dynamics (freq = 2)
mu_2 = results[-1][1][0]
delta_mu = mu_2 - mu_1

N_decoding = 5
plt.figure()
colors = matplotlib.cm.winter(np.linspace(0, 1, N_decoding))
for i, tt in enumerate(np.linspace(0, 1, N_decoding, endpoint=True)):
    z_sample = (mu_1 + tt * delta_mu).unsqueeze(0)
    p_y_pred, _, _ = model(t_context[0].unsqueeze(0),
                     x_context[0].unsqueeze(0),
                     t_target[0].unsqueeze(0),
                     z=z_sample)
    mu = p_y_pred.loc.detach()
    scale = p_y_pred.scale.detach()
    sorted_t, argsort = torch.sort(t_target.cpu()[0, :, 0])
    sorted_mu = mu.cpu()[0][argsort, :]
    plt.plot(sorted_t.numpy(),
                sorted_mu.numpy(),
                label=str(tt),
                color=colors[i]
             )
# plt.legend()
plt.savefig(save_folder + str(N_decoding) + 'interpolation_nolegend.png', dpi=800)


# Experiment 2: PCA in latent space
# We encode different time series dynamics context points into the
# latent space and use PCA to visualize the different mean encodings
# (mu_z).

# Note the strong separation between different dynamics.

k = 2  # number of principal components (2 for visualization)
A = torch.cat([A for freq, A, _ in results])
with torch.no_grad():
    U, S, V = torch.pca_lowrank(A)
    proj = torch.matmul(A, V[:, :k]).cpu().numpy()

colors = matplotlib.cm.winter(np.linspace(0, 1, len(freqs)))
plt.figure()
for i, freq in enumerate(freqs):
    plt.scatter(proj[i*100: (i+1)*100, 0], proj[i*100:(i+1)*100, 1], color=colors[i], label='freq= '+str(freq))
# plt.legend()
plt.savefig(save_folder + str(N_decoding) + 'pca_nolegend.png', dpi=800)

