# turbAE
Convolutional Autoencoder for e.g. two-dimensional turbulent fluid flows.

## Examples
- Two-dimensinal Rayleigh-Bénard convection at ${\rm Ra}= 10^6$ and ${\rm Pr}= 10$ in a rectangular box with aspect ratio $\Gamma = L_X/H = 4$. The boundary conditions were free-slip and constant temperature.

## Manual
For a tutorial see the prepared IPython Notebook `tutorial.ipynb` under /examples/two_dimensional_rbc/

## Introduction to Autoencoders
Autoencoders are feed-forward neural networks that take an input and map it back to itself. In the course of a classical Autoencoder network the input dimension is drastically reduced, yielding a reduced order representation of the original input.
This makes them a suitable tool for Reduced Order Modeling applications.

For a first introduction see e.g.:
- https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798
- https://neptune.ai/blog/representation-learning-with-autoencoder


## Requirements
Stable for
- `python `>=  3.6.0
- `torch`  >= 1.10.0
- `numpy`  >= 1.20.1
- `h5py`   >= 2.10.0 
- `pyyaml` >= 6.0
- `torchsummary` >= 1.5.1
