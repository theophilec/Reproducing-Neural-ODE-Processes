"""
Experiments for studying the learned mean embedding.

Two experiments:
    * PCA with several different dynamics.
    * Interpolating the latent space between two dynamics.
"""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from omegaconf import OmegaConf
import torch
from matplotlib.pyplot import rc as rc

from models import ConvNetModel, MLPModel
from datasets import RotNISTDataset, SineData
from utils import get_split
from torch.utils.data import DataLoader

sns.set_style('darkgrid')
sns.set_style('white')
rc('font', family='serif')
rc('text', usetex=True)
darkness = 0.15
truth_darkness = 1
color = '#9013FE'


def plot_sine_img(model, b, t, y, t_context, y_context, t_extra, save_folder, time_tag, num_lines_to_plot=25):
    fig = plt.figure()
    # for a batch element b
    for i in range(num_lines_to_plot):
        # plot a number of predictions (on t_extra)
        p_y_pred, _, _ = model(t_context, y_context, t_extra)
        mu = p_y_pred.loc
        order = np.argsort(t_extra[b].flatten().cpu().numpy())
        t_, y_ = t_extra[b].cpu().numpy()[order], mu[b].cpu().numpy()[order]
        if i == 0:
            plt.plot(t_, y_, alpha=darkness, c=color, zorder=-num_lines_to_plot,
                     label=r'prediction means $\mu_y(z)$ for $z \sim q(z|C)$')
        else:
            plt.plot(t_, y_, alpha=darkness, c=color, zorder=-num_lines_to_plot)
        plt.xlim(-3.14, 3.14)
        plt.ylim(-1.1, 1.1)
    # and plot the context and the full datapoints
    plt.plot(t[b].cpu().numpy(), y[b].cpu().numpy(), c='k', linestyle='--', alpha=truth_darkness, zorder=1,
             label='ground truth')
    plt.scatter(t_context[b].cpu().numpy(), y_context[b].cpu().numpy(), c='green', alpha=truth_darkness, zorder=2,
                label='context')
    plt.xlabel('t')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f'sine_img_plot_{time_tag}_{b}.png', dpi=300)
    plt.show()


def plot_rotnist_img(b, mu, T, t_target_rounded, t_target_initial, y_target, t_context, y_context, save_folder,
                     time_tag):
    # Extract mean of distribution
    im = mu[b].numpy().reshape(T, 28, 28)

    fig = plt.figure(figsize=(T, 3))
    for i, t_i in enumerate(t_target_rounded[b]):
        plt.subplot(3, T, i + 1)
        if t_i in t_target_initial[b]:
            index = torch.where(t_target_initial[b] == t_i)[0].item()
            plt.imshow(y_target[b, index].view(28, 28), cmap='gray')
        else:
            plt.imshow(np.zeros((28, 28)), cmap='gray')
        plt.axis('off')

        plt.subplot(3, T, i + T + 1)
        if t_i in t_context[b]:
            index = torch.where(t_context[b] == t_i)[0].item()
            plt.imshow(y_context[b, index].view(28, 28), cmap='gray')
        else:
            plt.imshow(np.zeros((28, 28)), cmap='gray')
        plt.axis('off')

        plt.subplot(3, T, i + 2 * T + 1)
        plt.imshow(im[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_folder / f'rotnist_test_{b}_{time_tag}.png', dpi=300)
    plt.show()


def plot_sine_figures(project_dir, time_tag, model_name):
    # make the save folder
    save_folder = project_dir / f'figures/{model_name}'
    try:
        os.makedirs(save_folder)
    except FileExistsError:
        pass

    # load previously trained model
    device = torch.device('cpu')
    conf = OmegaConf.load(project_dir / f'runs/sine/{model_name}/conf.yaml')
    h_sizes = OmegaConf.to_container(conf.hidden_sizes)
    model = MLPModel(dim_y=1, dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                     hidden_sizes_encoder=h_sizes,
                     hidden_sizes_ode_net=h_sizes,
                     hidden_sizes_decoder=h_sizes, t0=-0.1, device=device)
    model_weights = torch.load(project_dir / f'runs/sine/{model_name}/seed_0/model.pth',
                               map_location=device)
    model.load_state_dict(model_weights)


    # create a sine dataset
    dataset = SineData()
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, drop_last=True)
    t, y = next(iter(dataloader))
    t_context, y_context, t_extra, y_extra, _, _ = get_split(t, y, test_context_size=conf.test_context_size)
    with torch.no_grad():
        plot_sine_img(model, 0, t, y, t_context, y_context, t_extra, save_folder, time_tag)
        plot_sine_img(model, 1, t, y, t_context, y_context, t_extra, save_folder, time_tag)
        plot_sine_img(model, 2, t, y, t_context, y_context, t_extra, save_folder, time_tag)
        plot_sine_img(model, 3, t, y, t_context, y_context, t_extra, save_folder, time_tag)
        plot_sine_img(model, 4, t, y, t_context, y_context, t_extra, save_folder, time_tag)


def plot_rotnist_figures(project_dir, time_tag, model_name):
    # make the save folder
    save_folder = project_dir / f'figures/{model_name}'
    try:
        os.makedirs(save_folder)
    except FileExistsError:
        pass

    # load previously trained model
    device = torch.device('cpu')
    epoch = 140
    conf = OmegaConf.load(project_dir / f'runs/rotnist/{model_name}/conf.yaml')
    h_sizes = OmegaConf.to_container(conf.hidden_sizes)
    model = ConvNetModel(dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                         hidden_sizes_ode_net=h_sizes, t0=-0.1, device=device)
    model_weights = torch.load(project_dir / f'runs/rotnist/{model_name}/seed_0/model_ep{epoch}.pth',
                               map_location=device)
    model.load_state_dict(model_weights)

    # load RotNIST dataset, and get the test samples
    dataset_mnist = RotNISTDataset(data_dir=str(project_dir / 'data'))
    len_test = 10
    dataset_test = dataset_mnist[len(dataset_mnist) - len_test:]
    dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, drop_last=True)

    t_all, y_all = next(iter(dataloader_test))
    t_context, y_context, _, _, t_target_initial, y_target = get_split(t_all, y_all, context_range=(7, 8),
                                                                       extra_target_range=(9, 10))

    # Create a set of target points corresponding to entire [x_min, x_max] range
    extrapolation = 5
    t_min, t_max = 0., 1.5  # for the rotnist dataset
    t_target = torch.linspace(t_min, t_max + extrapolation / 10, 16 + extrapolation)
    t_target = t_target.view(1, -1, 1).repeat(t_context.shape[0], 1, 1)
    t_target_rounded = torch.round(t_target * 10 ** 3) / (10 ** 3)

    # get prediction on test samples, and plot the images for 2 examples
    p_y_pred, _, _ = model(t_context, y_context, t_target)
    mu = p_y_pred.loc.detach()
    _, T, _ = t_target.shape
    plot_rotnist_img(0, mu, T, t_target_rounded, t_target_initial, y_target, t_context, y_context, save_folder,
                     time_tag)
    plot_rotnist_img(9, mu, T, t_target_rounded, t_target_initial, y_target, t_context, y_context, save_folder,
                     time_tag)


def plot_sine_training_figures(project_dir, run_name):
    # make the save folder
    save_folder = project_dir / f'figures/{run_name}'
    try:
        os.makedirs(save_folder)
    except FileExistsError:
        pass

    df_mse_test = pd.read_csv(project_dir / f'runs/sine/{run_name}/aggregates/mse_test_epoch--{run_name}.csv',
                              delimiter=';').rename(columns={'Unnamed: 0': 'step'})[1:]
    df_mse_train = pd.read_csv(project_dir / f'runs/sine/{run_name}/aggregates/mse_train_epoch--{run_name}.csv',
                               delimiter=';').rename(columns={'Unnamed: 0': 'step'})[1:]
    df_loss = pd.read_csv(project_dir / f'runs/sine/{run_name}/aggregates/train_loss--{run_name}.csv',
                          delimiter=';').rename(columns={'Unnamed: 0': 'step'})[1:]

    plt.figure()
    plt.plot(df_mse_test['mean'], label='test MSE')
    plt.fill_between(df_mse_test.step, df_mse_test.amax, df_mse_test.amin, alpha=0.3)
    plt.plot(df_mse_train['mean'], label='train MSE')
    plt.fill_between(df_mse_train.step, df_mse_train.amax, df_mse_train.amin, alpha=0.3)
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f'sine_training.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    time_tag = datetime.now().strftime(f'%Y%m%d_%H%M%S')
    # plot_sine_figures(project_dir, time_tag, '10_full_T_20210222_114949')
    # plot_rotnist_figures(project_dir, time_tag, '10_full_T_20210222_174621')

    # 10_full_T_20210222_114949 also works for dim_l=10
    plot_sine_training_figures(project_dir, '2_full_T_20210222_115004')
