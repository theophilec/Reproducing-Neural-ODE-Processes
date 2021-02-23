from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import SineData, RotNISTDataset, FreqSineData, NoisySineData
from models import MLPModel, ConvNetModel
from utils import save_src, get_split, log_sine_plot, save_hparams, log_rotnist_plot2


def train(conf, project_dir: Path, run_dir: Path) -> torch.nn.Module:
    writer = SummaryWriter(str(run_dir))
    save_hparams(conf, writer)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if conf.dataset == 'sine':
        # dataset of time-series
        dataset_train = SineData()
        dataloader_train = DataLoader(dataset_train, batch_size=conf.batch_size, drop_last=True)
        dataset_test = SineData()
        dataloader_test = DataLoader(dataset_test, batch_size=conf.batch_size, shuffle=False, drop_last=True)

        h_sizes = OmegaConf.to_container(conf.hidden_sizes)  # OmegaConf object to list
        model = MLPModel(dim_y=1, dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                         hidden_sizes_encoder=h_sizes,
                         hidden_sizes_ode_net=h_sizes,
                         hidden_sizes_decoder=h_sizes,
                         t0=dataset_train.t0, device=device)
    elif conf.dataset == 'sinefreq':
        # dataset of frequency varying sinus time-series
        dataset_train = FreqSineData(amplitude_range=(0.5, 1.),
                                     shift_range=(-.5, .5),
                                     freq_range=(1.0, 2.0),
                                     num_samples=5000)
        dataloader_train = DataLoader(dataset_train, batch_size=conf.batch_size, drop_last=True)
        dataset_test = FreqSineData(amplitude_range=(0.5, 1.),
                                    shift_range=(-.5, .5),
                                    freq_range=(1.0, 2.0),
                                    num_samples=1000)
        dataloader_test = DataLoader(dataset_test, batch_size=conf.batch_size, shuffle=False, drop_last=True)

        h_sizes = OmegaConf.to_container(conf.hidden_sizes)  # OmegaConf object to list
        model = MLPModel(dim_y=1, dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                         hidden_sizes_encoder=h_sizes,
                         hidden_sizes_ode_net=h_sizes,
                         hidden_sizes_decoder=h_sizes,
                         t0=dataset_train.t0, device=device)
    elif conf.dataset == 'noisysine':
        sigma = conf.sigma
        # dataset of noisy sinus time-series
        dataset_train = NoisySineData(sigma,
                                      shift_range=(-0.1, .1),
                                      freq_range=(1.9, 2.0),
                                      num_samples=1000)
        dataloader_train = DataLoader(dataset_train, batch_size=conf.batch_size, drop_last=True)
        dataset_test = NoisySineData(sigma,
                                     shift_range=(-0.1, .1),
                                     freq_range=(1.9, 2.0),
                                     num_samples=1000)
        dataloader_test = DataLoader(dataset_test, batch_size=conf.batch_size, shuffle=False, drop_last=True)

        h_sizes = OmegaConf.to_container(conf.hidden_sizes)  # OmegaConf object to list
        model = MLPModel(dim_y=1, dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                         hidden_sizes_encoder=h_sizes,
                         hidden_sizes_ode_net=h_sizes,
                         hidden_sizes_decoder=h_sizes,
                         t0=dataset_train.t0, device=device)
    elif conf.dataset == 'rotnist':
        # dataset of Rotating MNIST (in the literature)
        dataset_mnist = RotNISTDataset(data_dir=str(project_dir / 'data'))
        len_test = 10
        dataset_train = dataset_mnist[:len(dataset_mnist) - len_test]
        dataset_test = dataset_mnist[len(dataset_mnist) - len_test:]
        dataloader_train = DataLoader(dataset_train, batch_size=conf.batch_size, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=conf.batch_size, shuffle=False, drop_last=True)
        h_sizes = OmegaConf.to_container(conf.hidden_sizes)
        model = ConvNetModel(dim_r=conf.dim_r, dim_z_prime=conf.dim_z_prime, dim_l=conf.dim_l,
                             hidden_sizes_ode_net=h_sizes, t0=dataset_mnist.t0, device=device)

    else:
        raise ValueError(f'Dataset {conf.dataset} not recognized')

    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=conf.lr)

    context_range = OmegaConf.to_container(conf.context_range)
    extra_target_range = OmegaConf.to_container(conf.extra_target_range)
    global_train_step = 0
    global_test_step = 0
    for epoch in tqdm(range(conf.epochs)):
        mse_train_list = []
        mse_test_list = []

        with torch.no_grad():
            for step, (t, y) in enumerate(dataloader_test):
                t, y = t.to(device), y.to(device)
                t_context, y_context, t_extra, y_extra, _, _ = get_split(t, y, test_context_size=conf.test_context_size)

                p_y, _, _ = model(t_context, y_context, t_extra)  # for testing, we only need predictions at t_extra
                output = p_y.loc
                mse_test = F.mse_loss(output, y_extra)

                # log test results
                writer.add_scalar('mse_test', mse_test.item(), global_test_step)
                mse_test_list.append(mse_test.item())
                if step == 0 and epoch % 2 == 0:
                    if conf.dataset in ['sine', 'sinefreq', 'noisysine']:
                        log_sine_plot(writer, model, t, y, t_context, y_context, t_extra, epoch)
                    elif conf.dataset == 'rotnist':
                        log_rotnist_plot2(writer, model, t, y, epoch, 'test')
                global_test_step += 1

        for (t, y) in dataloader_train:
            t, y = t.to(device), y.to(device)
            (
                t_context, y_context, t_extra, y_extra, t_target, y_target
            ) = get_split(t, y, context_range=context_range, extra_target_range=extra_target_range)

            p_y, q_z_T, q_z_C = model(t_context, y_context, t_target, y_target=y_target)
            log_p = p_y.log_prob(y_target).sum(dim=(1, 2)).mean(dim=0)  # mean on batch dim, sum on time dim/y dim

            output = p_y.loc
            mse_train = F.mse_loss(output, y_target)
            # mean on batch dim, sum on z dim (equivalent to kl_div of the multivariate normal)
            kl_div = kl_divergence(q_z_C, q_z_T).sum(dim=1).mean(dim=0)
            loss = - log_p + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log training metrics
            writer.add_scalar('kl_div', kl_div.item(), global_train_step)
            writer.add_scalar('log_p', log_p.item(), global_train_step)
            writer.add_scalar('train_loss', loss.item(), global_train_step)
            writer.add_scalar('mse_train', mse_train.item(), global_train_step)
            mse_train_list.append(mse_train.item())
            global_train_step += 1

        # log test/train mse epoch-wise to match the paper's figures
        writer.add_scalar('mse_train_epoch', np.mean(mse_train_list), epoch)
        writer.add_scalar('mse_test_epoch', np.mean(mse_test_list), epoch)
        if epoch % conf.checkpoint_freq == 0 and epoch > 0:
            torch.save(model.state_dict(), run_dir / f'model_ep{epoch}.pth')

    torch.save(model.state_dict(), run_dir / f'model.pth')
    return model


def main():
    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / 'src/conf.yaml')

    # Save and log config/code
    time_tag = datetime.now().strftime(f'%Y%m%d_%H%M%S')
    log_tag = f'{conf.dim_l}_full_T_{time_tag}'
    log_dir = project_dir / f'runs/{conf.dataset}/{log_tag}'
    save_src(log_dir, conf)

    # Make a single or multiple runs with this config
    print(OmegaConf.to_yaml(conf))
    for seed in range(conf.n_runs):
        seed_everything(seed)
        run_dir = log_dir / f'seed_{seed}'
        print(f'Training with seed {seed}...')
        train(conf, project_dir, run_dir)


if __name__ == '__main__':
    main()
