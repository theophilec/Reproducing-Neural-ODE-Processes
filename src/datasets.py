import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
from numpy import pi

# --- from the original code of the Neural ODE Processes paper
class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1
        self.t0 = -3.2

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-np.pi, np.pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class RotNISTDataset(Dataset):
    """
    Loads the rotated 3s from ODE2VAE paper
    https://www.dropbox.com/s/aw0rgwb3iwdd1zm/rot-mnist-3s.mat?dl=0
    """

    def __init__(self, data_dir):
        mat = loadmat(data_dir + '/rot-mnist-3s.mat')
        dataset = mat['X'][0]
        dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], -1)
        self.data = torch.tensor(dataset, dtype=torch.float32)
        self.t = (torch.arange(dataset.shape[1], dtype=torch.float32).view(-1, 1) / 10).repeat([dataset.shape[0], 1, 1])
        self.data = list(zip(self.t, self.data))
        self.t0 = float(self.t.min()) - 0.1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class NoisySineData(Dataset):
        """
        Dataset of functions f(x) = sin(w * x - b) + sigma * dB(x) where w and b and
        randomly sampled and dB is white noise.
        The function is evaluated from -pi to pi.

        Parameters
        ----------
        sigma : float
                Defines the noise level.

        shift_range : tuple of float
                Defines the range from which the shift (i.e. b) of the function is
                sampled.

        freq_range : tuple of float
                Defines the range from which the pulsation (i.e. w) of the
                function is sampled.

        num_samples : int
                Number of samples of the function contained in dataset.

        num_points : int
                Number of points at which to evaluate f(x) for x in [-pi, pi].
        """
        def __init__(self, sigma, shift_range=(-.5, .5),
                     freq_range=(0.3, 2.0), num_samples=1000, num_points=100):
            self.shift_range = shift_range
            self.num_samples = num_samples
            self.num_points = num_points
            self.x_dim = 1  # x and y dim are fixed for this dataset.
            self.y_dim = 1
            self.t0 = -3.2

            # Generate data
            self.data = []
            b_min, b_max = shift_range
            w_min, w_max = freq_range
            for i in range(num_samples):
                # Sample random shift
                b = (b_max - b_min) * np.random.rand() + b_min
                # Sample random frequency
                w = (w_max - w_min) * np.random.rand() + w_min
                # Shape (num_points, x_dim)
                x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
                # Shape (num_points, y_dim)
                y = torch.sin(w * x - b) + sigma * torch.randn(x.shape)
                self.data.append((x, y))

        def __getitem__(self, index):
                return self.data[index]

        def __len__(self):
                return self.num_samples


class FreqSineData(Dataset):
        """
        Dataset of functions f(x) = a * sin(w * x - b) where a, b, and w are
        randomly sampled. The function is evaluated from -pi to pi.

        Parameters
        ----------
        amplitude_range : tuple of float
                Defines the range from which the amplitude (i.e. a) of the function
                is sampled.

        shift_range : tuple of float
                Defines the range from which the shift (i.e. b) of the function is
                sampled.

        freq_range : tuple of float
                Defines the range from which the pulsation (i.e. w) of the
                function is sampled.

        num_samples : int
                Number of samples of the function contained in dataset.

        num_points : int
                Number of points at which to evaluate f(x) for x in [-pi, pi].
        """
        def __init__(self,
                     amplitude_range=(-1., 1.),
                     shift_range=(-.5, .5),
                     freq_range=(0.3, 2.0),
                     num_samples=1000,
                     num_points=100):
            self.amplitude_range = amplitude_range
            self.shift_range = shift_range
            self.num_samples = num_samples
            self.num_points = num_points
            self.x_dim = 1  # x and y dim are fixed for this dataset.
            self.y_dim = 1

            # Generate data
            self.data = []
            a_min, a_max = amplitude_range
            b_min, b_max = shift_range
            w_min, w_max = freq_range
            for i in range(num_samples):
                # Sample random amplitude
                a = (a_max - a_min) * np.random.rand() + a_min
                # Sample random shift
                b = (b_max - b_min) * np.random.rand() + b_min
                # Sample random frequency
                w = (w_max - w_min) * np.random.rand() + w_min
                # Shape (num_points, x_dim)
                x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
                # Shape (num_points, y_dim)
                y = a * torch.sin(w * x - b)
                self.data.append((x, y))

        def __getitem__(self, index):
                return self.data[index]

        def __len__(self):
                return self.num_samples
