import glob
import os
import shutil
from typing import Tuple

import torch
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
sns.set_style('darkgrid')
sns.set_style('white')
# plt.rc('font', family='serif')  # does not work on sorbonne machines
# plt.rc('text', usetex=True)
darkness = 0.35
truth_darkness = 1
color = '#9013FE'


def log_sine_plot(writer, model, t, y, t_context, y_context, t_extra, epoch, num_lines_to_plot=15):
    fig = plt.figure(figsize=(10, 6))
    for i in range(4):
        # for a batch element i
        ax = fig.add_subplot(2, 2, i + 1)
        for _ in range(num_lines_to_plot):
            # plot a number of predictions (on t_extra)
            p_y_pred, _, _ = model(t_context, y_context, t_extra)
            mu = p_y_pred.loc
            order = np.argsort(t_extra[i].flatten().cpu().numpy())
            t_, y_ = t_extra[i].cpu().numpy()[order], mu[i].cpu().numpy()[order]
            ax.plot(t_, y_, alpha=darkness, c=color, zorder=-num_lines_to_plot)
            ax.set_xlim(-3.14, 3.14)
            ax.set_ylim(-1.1, 1.1)
        # and plot the context and the full datapoints
        ax.plot(t[i].cpu().numpy(), y[i].cpu().numpy(), c='k', linestyle='--', alpha=truth_darkness, zorder=1)
        ax.scatter(t_context[i].cpu().numpy(), y_context[i].cpu().numpy(), c='k', alpha=truth_darkness, zorder=2,
                   label='context')
        ax.set_xlabel('t')
    plt.legend()
    plt.tight_layout()
    writer.add_figure('sine_plot', fig, global_step=epoch)


def log_rotnist_plot2(writer, model, t_all, y_all, epoch, tag):
    t_context, y_context, _, _, t_target_initial, y_target = get_split(t_all, y_all, context_range=(6, 7),
                                                                       extra_target_range=(7, 8))
    # Create a set of target points corresponding to entire [x_min, x_max] range
    extrapolation = 5
    t_min, t_max = 0., 1.5  # for the rotnist dataset
    t_target = torch.linspace(t_min, t_max + extrapolation / 10, 16 + extrapolation, device=model.device)
    t_target = t_target.view(1, -1, 1).repeat(t_context.shape[0], 1, 1)
    t_target_rounded = torch.round(t_target * 10 ** 3) / (10 ** 3)

    p_y_pred, _, _ = model(t_context, y_context, t_target)
    mu = p_y_pred.loc.detach()
    _, T, _ = t_target.shape
    for b in range(4):
        # Extract mean of distribution
        im = mu[b].cpu().numpy().reshape(T, 28, 28)

        fig = plt.figure(figsize=(T, 3))
        for i, t_i in enumerate(t_target_rounded[b]):
            plt.subplot(3, T, i + 1)
            if t_i in t_target_initial[b]:
                index = torch.where(t_target_initial[b] == t_i)[0].item()
                plt.imshow(y_target[b, index].view(28, 28).cpu(), cmap='gray')
            else:
                plt.imshow(np.zeros((28, 28)), cmap='gray')
            plt.axis('off')

            plt.subplot(3, T, i + T + 1)
            if t_i in t_context[b]:
                index = torch.where(t_context[b] == t_i)[0].item()
                plt.imshow(y_context[b, index].view(28, 28).cpu(), cmap='gray')
            else:
                plt.imshow(np.zeros((28, 28)), cmap='gray')
            plt.axis('off')

            plt.subplot(3, T, i + 2 * T + 1)
            plt.imshow(im[i], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        writer.add_figure(f'{tag}_img_plot_v2_{b}', fig, global_step=epoch)


def save_src(log_dir: str, conf):
    os.makedirs(log_dir, exist_ok=True)
    # save config and source files as text files
    with open(f'{log_dir}/conf.yaml', 'w') as f:
        OmegaConf.save(conf, f)
    for f in glob.iglob('*.py'):
        shutil.copy2(f, log_dir)


def save_hparams(conf, writer):
    conf_clean = {k: str(v) for (k, v) in conf.items()}
    writer.add_hparams(conf_clean, metric_dict={'score': 0.})


def get_split(t, y, context_range: Tuple = None, extra_target_range: Tuple = None, test_context_size: int = None):
    N, num_points, _ = t.shape

    if test_context_size is not None:
        num_context = test_context_size
        num_extra_target = num_points - num_context
    else:
        num_context = np.random.randint(*context_range)
        num_extra_target = np.random.randint(*extra_target_range)

    locations = np.random.choice(num_points, size=num_context + num_extra_target, replace=False)
    t_context = t[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    t_extra = t[:, locations[num_context:], :]
    y_extra = y[:, locations[num_context:], :]
    t_target = t[:, locations, :]  # (N, num_context + num_extra_target, 1)
    y_target = y[:, locations, :]  # (N, num_context + num_extra_target, dim_y)

    return t_context, y_context, t_extra, y_extra, t_target, y_target


def context_target_split(x, y, num_context, num_extra_target, locations=None):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    if locations is None:
        locations = np.random.choice(num_points,
                                     size=num_context + num_extra_target,
                                     replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target
