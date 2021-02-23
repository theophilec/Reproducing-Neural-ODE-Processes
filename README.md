# Reproducing the results in _Neural ODE Processes_ (ICLR 2021)

This repo presents code to reproduce the results in the ICLR 2021 paper _Neural ODE Processes_ by Alexander Norcliffe, Cristian Bodnar, Ben Day, Jacob Moss, and Pietro Liòrepo (https://openreview.net/forum?id=27acGyyI1BY).

This reproduction is an independent initiative by Matthias Cremieux, Alexandre Thomas, and Théophile Cantelobre (as part of our Advanced Machine Learning course at Sorbonne Université). However, we did use certain parts of the code the authors released (in particular with respect to dataset creation and plotting).

## Highlights

Insert a few images !

If you have any questions, feel free to get into touch with us at first_name.last_name@mines-paristech.fr

## Code structure

### Dependencies

The code uses the `torch` library as well as other common Python machine learning libraries.

### Training a model

To train a model, set parameters in `conf.yaml` (dataset, number of epochs, seed, learning rate, ...) and run `main.py`. Training in logged in `tensorboard` in the `runs` directory.

## Details

Differences between the original implementation and ours :

- Unlike their implementation, during test time we only predict values for the extra target points (points in the target
  set but not in the context set) instead of considering the whole context set
    - for the sine dataset, the testing context set is made of 10/100 examples, and extra target is the other 90/100
    - therefore, for the sine dataset, we use (1,5) as the range for the number of extra target points instead of (0,5)
      in their code, in order to ensure we have an extra target point
- specific to rotating MNIST task :
    - for the ConvNDP, slightly change the decoder convolution shapes to output 28x28 instead of 32x32 (original
      implementation `img_regression.py` does not run)
    - use 20% of the dataset (209 examples) for testing, instead of 10 examples
