# Stochastic Wasserstein Barycenters

Code accompanying [https://arxiv.org/abs/1802.05757](https://arxiv.org/abs/1802.05757)

## Description
Optimal transport (OT) defines a powerful way to compare and transform distributions using a metric on the space of probability distributions given by the Wasserstein distance. In particular, Wasserstein barycenters can be understood as Frechet means in the space of probability distributions.

We propose a new stochastic algorithm to compute the barycenter of a set of input distributions (either discrete or continuous) that only depends on the ability to sample from the input distributions and does not require a discretization of the support of the barycenter.

## Codebase
There are two implementations of the paper in this repository.

### MATLAB
A MATLAB version meant as a proof of concept. The MATLAB code is fully commented, easier to understand, and has no dependencies. It is not intended for large applications. We include an example `test.m` script for typical usage.

### C++
A C++ version that we have used to generate the results in the paper. The C++ version depends on Eigen and LBFGS which are locally included, as well as OpenCV for image processing. This version of the code has much sparser documentation.

To compile the C++ version:
```
cd c++/
mkdir build
cd build/
cmake ..
make
```

An example usage is `./main --folder figs/ --gaussian 3` which computes the Wasserstein barycenter of three random Gaussian distributions.
