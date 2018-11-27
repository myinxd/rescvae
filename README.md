# ResCVAE
This repo aims to construct a radio galaxy morphology generator by training a residual convolutional neural blocks constructed autoencoder (CVAE) with variational inference, and simulate new samples by feeding randomly generated vectors into the decoder subnet.

## Construction of the package
In summary, we design classes for constructing the ResCVAE network, as well as some utilities for image preprocessing, network saving and restoration, and etc. The users can build their own network with mean squared error (MSE), or cross entropy (CE) loss functions to form the reconstruction objective $$\mathcal{P}(\mathrm{X}|z)$$. For detailed instruction and usage, please refer to the code files and our [paper](https://github.com/myinxd/rescvae/document/paper-rescvae.pdf)<TODO>.

#### Demos on MNIST
Notebooks of deploying the ResCVAE on the MNIST dataset are provided as examples for the users to construct their own ResCVAE network, which are 
- [notebook-rescvae-mnist](https://github.com/myinxd/rescvae/blob/master/demo-mnist/notebook-rescvae-mnist.ipynb): Train a ResCVAE network for mnist hand-written digits simulation;
- [notebook-rescvae-mnist-generation](https://github.com/myinxd/rescvae/blob/master/demo-mnist/notebook-cvae-mnist-generation.ipynb): Generate new handwritten images.

#### Demos on radio galaxy generation
We also provide the notebooks to on radio galaxy image generation task [here](https://github.com/myinxd/rescvae/blob/master/demo-radiogalaxy),as well as the trained model. For the detail usage please refer to the codes.

## Requirements
Some python packages are required before applying the nets, which are listed as follows. A [setup](https://github.com/myinxd/rescvae/blob/master/setup.py) file is provided to automatically install the packages.
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [astropy](https://www.astropy.org/)
- [matplotlib](http://www.matplotlib.org/)
- [tensorflow-gpu](http://www.tensorflow.org/) or [tensorflow-cpu](http://www.tensorflow.org)
- [scikit-learn](http://scikit-learn.org/)
- [requests](http://www.python-requests.org/en/master/)

Also, [CUDA](http://develop.nvidia.org/cuda) is required if you want to run the codes by GPU, a Chinese guide for CUDA installation on Ubuntu 16.04 is [here](http://www.mazhixian.me/2017/12/13/Install-tensorflow-with-gpu-library-CUDA-on-Ubuntu-16-04-x64/). Since the memory-require issue, we advise a NVIDIA GTX 750 above GPU card for running our model.

## Usage
Before constructing a ResCVAE net, we would like to ask the users to install the package. Here is the installation steps,
```sh
$ cd rg-cvae
$ pip3 install <--user> <-e> .
```
You may use `-e` for editable and `--user` for user only.

Detailed usage of our rg-cvae package is demonstrated in [demo-mnist](https://github.com/myinxd/rg-cvae/blob/master/demo/demo-mnist/) and [demo-rg](https://github.com/myinxd/rg-cvae/blob/master/demo/demo-rg/) by jupyter notebooks. Below are examples of handwritten digits and radio galaxies.

- MNIST
<center>
<img src="https://github.com/myinxd/rescvae/blob/master/demo/demo-mnist/mnist_generated.png?raw=true" height=500 width=500>
</center>

- Radio galaxy
<center>
<img src="https://github.com/myinxd/rescvae/blob/master/demo/demo-rg/rg_generated.png?raw=true" height=100 width=500>
</center>

## Contributor
- Zhixian MA <`zx at mazhixian.me`>

## Citation
<TODO>

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.