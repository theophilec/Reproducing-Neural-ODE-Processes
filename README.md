# Reproducing the results in _Neural ODE Processes_ (ICLR 2021)

This repo presents code to reproduce the results in the ICLR 2021 paper _Neural ODE Processes_ by Alexander Norcliffe, Cristian Bodnar, Ben Day, Jacob Moss, and Pietro Liòrepo (https://openreview.net/forum?id=27acGyyI1BY).

This reproduction is an independent initiative by Matthias Cremieux, Alexandre Thomas, and Théophile Cantelobre (as part of our Advanced Machine Learning course at Sorbonne Université). However, we did use certain parts of the code the authors released (in particular with respect to dataset creation and plotting).

If you have any questions, feel free to get into touch with us at first_name.last_name@mines-paristech.fr

## Highlights

Model predictions on an example from the sine dataset, after training, and MSE curves during training (on the sine dataset, with default hyper-parameters) :

<img src="img/sine.png" alt="sine" style="zoom:24%;" /><img src="img/sine_training.png" style="zoom:24%;" />

Model predictions (bottom row) on an example of the RotNIST dataset. Top row contains all target points, and mid row contains context points :

<img src="img/rotnist.png" alt="rotnist" style="zoom:9%;" />

## Code structure

### Dependencies

The code uses the `torch` library as well as other common Python machine learning libraries (including `torchdiffeq` for the ODE part).

### Data

The RotNIST dataset can be downloaded [here](https://www.dropbox.com/s/aw0rgwb3iwdd1zm/rot-mnist-3s.mat), and placed inside the `data` folder, all others datasets (sine, etc) are simulated.

### Training a model

To train a model, set parameters in `conf.yaml` (dataset, number of epochs, seed, learning rate, ...) and run `main.py`. Training logs (loss, mse, images, model checkpoints) are located in the `runs` directory and can be visualised with the command `tensorboard --logdir runs`.

## Details

Differences between the original implementation (code submitted by the authors as supplementary material, *which was still in an experimental stage*) and ours :

- During test time we only predict values for the extra target points (points in the target set but not in the context set) instead of considering the whole context set
- specific to the sine dataset : we use (1,5) as the range for the number of extra target points instead of (0,5) in their code, in order to ensure we have at least one extra target point
- specific to rotating MNIST task : for the `ConvNDP`, slightly change the decoder convolution shapes to output 28x28 instead of 32x32, to match RotNIST image dimensions
