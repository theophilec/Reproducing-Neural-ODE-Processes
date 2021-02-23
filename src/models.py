from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from torchdiffeq import odeint


class MLP(nn.Module):
    def __init__(self, dim_in: int, hidden_sizes: List[int], dim_out: int, activation: str, last_activation=None):
        super().__init__()

        # hidden layers
        sizes = [dim_in] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1])]
            if activation == 'relu':
                layers += [nn.ReLU()]
            elif activation == 'tanh':
                layers += [nn.Tanh()]
            else:
                raise ValueError(f'activation {activation} not valid')

        # last layer
        layers += [nn.Linear(sizes[-1], dim_out)]
        if last_activation == 'relu':
            layers += [nn.ReLU()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Model(nn.Module, ABC):
    def __init__(self, dim_r: int, dim_z_prime: int, dim_l: int, hidden_sizes_ode_net: List[int], t0: float,
                 device: torch.device):
        super().__init__()
        self.linear_r_to_h = nn.Linear(dim_r, dim_r)  # context representation r to h
        self.linear_mu_z = nn.Linear(dim_r, dim_l + dim_z_prime)
        self.linear_sigma_z = nn.Linear(dim_r, dim_l + dim_z_prime)
        self.ode_mlp = MLP(dim_l + dim_z_prime + 1, hidden_sizes_ode_net, dim_l, activation='tanh')

        self.dim_l = dim_l
        self.dim_z_prime = dim_z_prime
        # t0 is minimum possible value of t (and depends on the dataset)
        self.t0 = t0
        self.device = device

    @abstractmethod
    def encoder(self, t, y) -> torch.Tensor:
        """
        Parameters
        ----------
        t: (N, T, 1)
        y: (N, T, dim_y)

        Returns
        -------
        Tensor r of shape (N, T, dim_r), the encoded representation of the context/target set (t, y)
        """
        pass

    @abstractmethod
    def decoder(self, t, latent_states, z_prime) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        t: (N, T, 1)
        latent_states: (N, T, dim_l)
        z_prime: (N, T, dim_z_prime)

        Returns
        -------
        Tensors mu and sigma of shape (N, T, dim_y), such that p(y_t|g(l_t,t)) = N(mu, diag(sigma))
        """
        pass

    def q(self, t: torch.Tensor, y: torch.Tensor):
        # t (N, T, 1) and y (N, T, dim_y), can be context set C or target set T

        # encode and aggregate
        r = self.encoder(t, y)  # (N, T, dim_r)
        r = r.mean(dim=1)  # (N, dim_r)

        # context representation r to parameters of q(z|C) : mu_z(r), sigma_z(r)
        h = self.linear_r_to_h(r)  # (N, dim_r)
        mu_z = self.linear_mu_z(h)  # (N, dim_z)
        assert not mu_z.isnan().any()
        sigma_z = self.linear_sigma_z(h)  # (N, dim_z)
        sigma_z = 0.1 + 0.9 * torch.sigmoid(sigma_z)
        return Normal(mu_z, sigma_z)

    def p(self, t: torch.Tensor, latent_states: torch.Tensor, z_prime: torch.Tensor):
        # t (N, T, 1), latent_states (N, T, dim_l)
        mu_y, sigma_y = self.decoder(t, latent_states, z_prime)  # (N, T, dim_y)
        sigma_y = 0.1 + 0.9 * torch.nn.functional.softplus(sigma_y)
        return Normal(mu_y, sigma_y)

    def ode_func(self, t, v):
        N, _ = v.shape  # (N, dim_l + dim_z_prime)
        t = t.view(1, 1).repeat(N, 1).to(self.device)  # t is already a tensor, make it of shape (N, 1)
        dl = self.ode_mlp(torch.cat([v, t], dim=1))  # (N, dim_l)
        dz_prime = torch.zeros(N, self.dim_z_prime,
                               device=self.device)  # no variations in z_prime, our constant encoded context
        assert not dl.isnan().any()
        return torch.cat([dl, dz_prime], dim=1)  # return dv

    def forward(self, t_context: torch.Tensor, y_context: torch.Tensor, t_target: torch.Tensor,
                y_target: torch.Tensor = None, z: torch.Tensor = None):
        """

        Parameters
        ----------
        t_context: (batch_size, num_context_points, 1)
        y_context: (batch_size, num_context_points, dim_y)
        t_target: (batch_size, num_target_points, 1)
        y_target: (batch_size, num_target_points, dim_y)
            Optional and only specified during training, in this case z is sampled from q(z|T)
        z: (batch_size, dim_z)
            Optional, if specified do not sample from q(z|C) or q(z|T) but use this z

        Returns
        -------
            p_y, q_z_T, q_z_C
                with q_z_T=None in the case y_target=None, and p_y the predictions at points t_target
        """
        batch_size = t_context.shape[0]
        t0 = torch.tensor(self.t0, device=self.device).view(1, 1, 1).repeat(batch_size, 1, 1)
        # encode target/context sets and sample context
        q_z_C = self.q(t_context, y_context)
        q_z_T = None

        if z is None:
            if y_target is None:
                # during testing, we don't have access to the target set, and we sample from the context set
                z = q_z_C.rsample()
            else:
                # during training, we need q_z_T to compute the loss and we sample from the whole target set
                q_z_T = self.q(t_target, y_target)
                z = q_z_T.rsample()  # z = [l(0), z_prime], of shape (N, dim_l + dim_z_prime)

        # integrate to get latent states at prediction times
        # shapes :
        #   t_target_sorted (num_unique_points,)
        #   t_target_indices (N, num_target + 1, 1) same shape as input
        t_target_sorted, t_target_indices = torch.unique(torch.cat([t0, t_target], dim=1), sorted=True,
                                                         return_inverse=True)
        # v is of shape (N, T', dim_l + dim_z_prime) with :
        #   v_t = [l(t), z_prime]
        #   T' = num_unique_points < T = num_target if duplicates
        v = odeint(self.ode_func, z, t_target_sorted)  # (T', N, dim_v)
        # todo: use odeint_adjoint? (more stable numerically)
        v = v.permute(1, 0, 2)  # (N, T', dim_v)
        t_target_indices = t_target_indices.repeat(1, 1, self.dim_l)  # (N, T+1, dim_l)
        latent = v.gather(dim=1, index=t_target_indices)  # (N, T+1, dim_l), get the initial order
        latent = latent[:, 1:, :]  # we don't care about l_0

        z_prime = z[:, self.dim_l:]
        p_y = self.p(t_target, latent, z_prime)  # distrib of shape (N, num_target, dim_y)
        return p_y, q_z_T, q_z_C


class MLPModel(Model):
    def __init__(self, dim_y: int, dim_r: int, dim_z_prime: int, dim_l: int,
                 hidden_sizes_encoder: List[int], hidden_sizes_ode_net: List[int],
                 hidden_sizes_decoder: List[int], t0: float, device: torch.device):
        super(MLPModel, self).__init__(dim_z_prime=dim_z_prime, hidden_sizes_ode_net=hidden_sizes_ode_net, t0=t0,
                                       dim_l=dim_l, dim_r=dim_r, device=device)
        self.encoder_mlp = MLP(dim_y + 1, hidden_sizes_encoder, dim_r, activation='relu')

        dim_h = hidden_sizes_decoder[-1]  # size of the hidden layer coming from xlz_to_hidden, before mu_y/sigma_y
        self.decoder_mlp = MLP(dim_l + 1, hidden_sizes_decoder, dim_h, activation='relu', last_activation='relu')
        self.decoder_mu = nn.Linear(dim_h + dim_l, dim_y)
        self.decoder_sigma = nn.Linear(dim_h + dim_l, dim_y)

    def encoder(self, t, y) -> torch.Tensor:
        return self.encoder_mlp(torch.cat([t, y], dim=2))

    def decoder(self, t, latent_states, z_prime) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder_mlp(torch.cat([t, latent_states], dim=2))
        mu = self.decoder_mu(torch.cat([h, latent_states], dim=2))
        pre_sigma = self.decoder_sigma(torch.cat([h, latent_states], dim=2))
        return mu, pre_sigma


class ConvNetModel(Model):
    def __init__(self, dim_r: int, dim_z_prime: int, dim_l: int, hidden_sizes_ode_net: List[int], t0: float,
                 device: torch.device):
        super().__init__(dim_z_prime=dim_z_prime, hidden_sizes_ode_net=hidden_sizes_ode_net, t0=t0, dim_l=dim_l,
                         dim_r=dim_r, device=device)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=(2, 2)),
            nn.ReLU(),
        )
        self.encoder_mlp = MLP(dim_in=128 * 2 * 2 + 128, hidden_sizes=[500], dim_out=dim_r, activation='relu')

        self.decoder_linear = nn.Linear(dim_l + dim_z_prime + 1, 4 * 4 * 8)  # (l, z', t) -> cnn_input_t (c=8, w=4, h=4)
        self.decoder_cnn = nn.Sequential(
            # output shape (C=128, W=7, H=7)
            nn.ConvTranspose2d(in_channels=8, out_channels=128, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (64, 14, 14)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=(2, 2),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (32, 28, 28)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=(2, 2),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder_to_sigma = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=(2, 2))
        self.decoder_to_mu = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=(2, 2))

    def encoder(self, t, y) -> torch.Tensor:
        N, num_points, dim_y = y.shape
        assert dim_y == 784
        y = y.view(N * num_points, 1, 28, 28)  # conv2D requires (N, C, H, W) inputs
        y = self.encoder_cnn(y)  # (N * num_points, 128, 2, 2)
        y = y.view(N, num_points, -1)
        t = t.repeat(1, 1, 128)  # why ? in order to give more importance to the time component ?
        r = self.encoder_mlp(torch.cat([y, t], dim=2))  # (N, num_points, dim_r)
        return r

    def decoder(self, t, latent_states, z_prime) -> Tuple[torch.Tensor, torch.Tensor]:
        N, num_points, _ = t.shape
        z_prime = z_prime.view(N, 1, -1).repeat(1, num_points, 1)  # (N, num_points, dim_z_prime)
        x = torch.cat([t, latent_states, z_prime], dim=2)
        x = self.decoder_linear(x)
        x = x.view(N * num_points, 8, 4, 4)  # convTranspose2D requires (N, C, H, W) inputs
        x = self.decoder_cnn(x)
        pre_sigma = self.decoder_to_sigma(x)
        mu = self.decoder_to_mu(x)
        pre_sigma = pre_sigma.view(N, num_points, 784)
        mu = mu.view(N, num_points, 784)
        return mu, pre_sigma
