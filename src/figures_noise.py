"""
Experiments for studying impact of noise.

Forked from ndp/plotting_1d_gif.py
"""

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


from models import MLPModel
sns.set_style('darkgrid')
sns.set_style('white')
rc('font', family='serif')
rc('text', usetex=True)

np.random.seed(42069)
torch.random.manual_seed(45)
#########################
#Change these
dataname = 'noisysine'
sigma = 0.2 # for noisysine
process_choice = 'nodep'
experiment_load = '1'
length_context = 20
start_point = 2
darkness = 0.10
truth_darkness = 1
colour = '#9013FE'
number_to_plot = 35
#########################

#model_name ='10_full_T_20210222_191700'
model_name = '10_full_T_20210222_184759' # for 0.2
# make the save folder
save_folder = '../figures/'
try:
    os.makedirs('./'+save_folder)
except FileExistsError:
    pass

device = torch.device('cpu')

conf = OmegaConf.load("../runs/" + dataname + '/' + model_name +'/conf.yaml')
h_sizes = OmegaConf.to_container(conf.hidden_sizes)
model = MLPModel(dim_y=1, dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                 hidden_sizes_encoder=h_sizes,
                 hidden_sizes_ode_net=h_sizes,
                 hidden_sizes_decoder=h_sizes,
                 t0=-3.2,
                 device=device)
model.load_state_dict(torch.load('../runs'  + '/' + dataname+ '/' +  model_name + '/seed_0/model_ep10.pth'))


#import data
from datasets import NoisySineData
from math import pi
dataset = NoisySineData(sigma,
                        shift_range=(-0.00001, .00001),
                        freq_range=(1.9999999, 2.0),
                        num_samples=10)
x_min, x_max = -pi, pi
y_min, y_max = -1.1, 1.1
initial_x = -3.2

t = dataset[0][0]
x = dataset[0][1]

x_true = np.sin(2*t)

# choosing the context points
t_c = []
x_c = []

choices = np.random.choice(np.arange(len(x)), size=length_context, replace=False)
choices = np.sort(choices)

for i in range(length_context):
    t_c.append(t[choices[i]])
    x_c.append(x[choices[i]])

t_full_context = torch.tensor(t_c).float().reshape((1, length_context, 1))
x_full_context = torch.tensor(x_c).float().reshape((1, length_context, 1))
t_target = t.reshape((1, len(t), 1))



from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
frame = 15
fig, ax = plt.subplots(figsize=(5,4))
t_context = t_full_context[:, :frame, :].to(device)
x_context = x_full_context[:, :frame, :].to(device)
for _ in range(number_to_plot):
    # Neural process returns distribution over y_target
    p_y_pred, q_z_T, q_z_C = model(t_context, x_context, t_target)
    # Extract mean of distribution
    mu = p_y_pred.loc.detach()
    scale = p_y_pred.scale.detach()
    plt.plot(t_target.cpu().numpy()[0], mu.cpu().numpy()[0], alpha=darkness, c=colour, zorder=-number_to_plot)
ax.plot(t, x_true, c='k', alpha=truth_darkness, zorder=1)
ax.plot(t, x_true + scale.cpu().numpy()[0], linestyle=":",alpha=truth_darkness, zorder=1, color="xkcd:black")
ax.plot(t, x_true - scale.cpu().numpy()[0],linestyle=":", alpha=truth_darkness, zorder=1, color="xkcd:black")
ax.plot(t, x_true + sigma * np.ones_like(x_true), linestyle="--", alpha=0.5, zorder=1, color="xkcd:blue")
ax.plot(t, x_true - sigma * np.ones_like(x_true), linestyle="--", zorder=1, alpha=0.5, color="xkcd:blue")
ax.scatter(t_context[0].cpu().numpy(), x_context[0].cpu().numpy(), c='green', alpha=truth_darkness, zorder=2)
axins = zoomed_inset_axes(ax, 3, loc="lower right")
axins.set_xlim(-3, -2)
axins.set_ylim(.5, 1.1)
axins.plot(t, x_true, c='k', alpha=truth_darkness, zorder=1)
axins.plot(t, x_true + scale.cpu().numpy()[0], linestyle=":",alpha=truth_darkness, zorder=1, color="xkcd:black")
axins.plot(t, x_true - scale.cpu().numpy()[0],linestyle=":", alpha=truth_darkness, zorder=1, color="xkcd:black")
axins.plot(t, x_true + sigma * np.ones_like(x_true), linestyle="--", alpha=0.5, zorder=1, color="xkcd:blue")
axins.plot(t, x_true - sigma * np.ones_like(x_true), linestyle="--", zorder=1, alpha=0.5, color="xkcd:blue")
axins.scatter(t_context[0].cpu().numpy(), x_context[0].cpu().numpy(), c='green', alpha=truth_darkness, zorder=2)
for _ in range(number_to_plot):
    # Neural process returns distribution over y_target
    p_y_pred, q_z_T, q_z_C = model(t_context, x_context, t_target)
    # Extract mean of distribution
    mu = p_y_pred.loc.detach()
    scale = p_y_pred.scale.detach()
    axins.plot(t_target.cpu().numpy()[0], mu.cpu().numpy()[0], alpha=darkness, c=colour, zorder=-number_to_plot)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.draw()
plt.savefig(save_folder+str(sigma)+'inset.png', dpi=800)
plt.cla()
