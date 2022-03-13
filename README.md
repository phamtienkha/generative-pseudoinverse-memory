# Generative Pseudo-Inverse Memory

Implementation of ICLR 2022 paper Generative Pseudo-Inverse Memory. The full paper can be found at https://openreview.net/forum?id=Harn4_EZBw. 

## Abstract

We propose Generative Pseudo-Inverse Memory (GPM), a class of deep generative memory models that are fast to write in and read out. Memory operations are recast as seeking robust solutions of linear systems, which naturally lead to the use of matrix pseudo-inverses. The pseudo-inverses are iteratively approximated, with practical computation complexity of almost O(1). We prove theoretically and verify empirically that our model can retrieve exactly what have been written to the memory under mild conditions. A key capability of GPM is iterative reading, during which the attractor dynamics towards fixed points are enabled, allowing the model to iteratively improve sample quality in denoising and generating. More impressively, GPM can store a large amount of data while maintaining key abilities of accurate retrieving of stored patterns, denoising of corrupted data and generating novel samples. Empirically we demonstrate the efficiency and versatility of GPM on a comprehensive suite of experiments involving binarized MNIST, binarized Omniglot, FashionMNIST, CIFAR10 & CIFAR100 and CelebA.

## Model Architecture

![gpm](https://user-images.githubusercontent.com/16438545/158054853-871b33e6-f36e-4644-87c0-2a9e719effe1.svg)

## Experiments

We report the negative evidence lower bound of test likelihood (lower is better) in the below table of GPM and some related memory models. Results are in nats/images for binarized datasets and bits/dim for RGB dataset. 

| Method      | Binarized MNIST | Binarized Omniglot  | CIFAR 10
| :---       |    :----:   |     :---:     | :---: |
| Kanerva Machine [1]      | -       | 68.3   | 4.37 |
| Dynamic Kanerva Machine [2]   | 75.3        | 77.2     | 4.79 |
| Kanerva++ [3] | 41.58 | 66.24 | **3.28** |
| Generative Pseudo-Inverse Memory (ours)| **31.48** | **25.68** | 4.03 | 

## References

[1] Yan Wu, Greg Wayne, A. Graves, and T. Lillicrap. The kanerva machine: A generative distributed memory. ICLR 2018.

[2] Yan Wu, Greg Wayne, Karol Gregor, and Timothy Lillicrap. Learning attractor dynamics for generative memory. NeurIPS 2018.

[3] Jason Ramapuram, Yan Wu, and Alexandros Kalousis. Kanerva++: extending the kanerva machine with differentiable, locally block allocated latent memory. ICLR 2021.

