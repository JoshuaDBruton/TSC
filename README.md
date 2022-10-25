# Translated Skip Connections: Expanding the Receptive Fields of Fully Convolutional Neural Networks
<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/JoshuaDBruton/TSC)](https://github.com/JoshuaDBruton/TSC/issues)
[![GitHub forks](https://img.shields.io/github/forks/JoshuaDBruton/TSC)](https://github.com/JoshuaDBruton/TSC/network)
[![GitHub stars](https://img.shields.io/github/stars/JoshuaDBruton/TSC)](https://github.com/JoshuaDBruton/TSC/stargazers)
[![GitHub license](https://img.shields.io/github/license/JoshuaDBruton/TSC)](https://github.com/JoshuaDBruton/TSC/blob/main/LICENSE)

</div>

## Overview
Here we present a novel neural network module, called a Translated Skip Connection, that can exponentially increase the receptive fields of neural networks with minimal impact on other design decisions. This work was completed as a part of my MSc. in Computer Science at the University of the Witwatersrand.
We have produced a paper for this work and are currently in the submission/review process with a journal. This README will be updated with details should the work pass review for publication.

Authored by: Joshua Bruton  
Supervised by: Dr. Hairong Wang  

## Contents
This repository contains implementations or usage of the following techniques and architectures:
1. Implementations of UNet, BNet (a smaller version of VNet), and TSCNet
2. Translated Skip Connections, used in the TSCNet architecture (marked with comment in TSCNet.py)
3. Optional Translational Equivariance (marked with comment in experiments/dataset/image_pair.py)
4. Dice Loss

We make use of Pytorch with Pytorch Lightning for our implementations.

## Usage
I have created a requirements file. I recommend using [pipenv](https://pypi.org/project/pipenv/) with Python 3.8 to open a shell and then using
~~~
pipenv install -r requirements.txt
~~~
and requirements should be met. Of course, Conda, and any other environment manager you are familiar with will work as well.

## Future work
This repository is licensed under the GNU General Public License and therefore is completely free to use for any project you see fit. If you do use or learn from our work, we would appreciate a citation, we will make the details available here after publication as this work is still in the review process.

## Suggestions
If there are any pressing problems with the code please open an issue and I will attend to it as timeously as is possible.

## Citation
~~~
@inproceedings{bruton2022translated,
  title={Translated Skip Connections-Expanding the Receptive Fields of Fully Convolutional Neural Networks},
  author={Bruton, J and Wang, H},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={631--635},
  year={2022},
  organization={IEEE}
}
~~~
