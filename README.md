# Translated Skip Connections: Expanding the Receptive Fields of Fully Convolutional Neural Networks
## Overview
Here we present a novel neural network module, called a Translated Skip Connection, that can exponentially increase the receptive fields of neural networks with minimal impact on other design decisions. This work was completed as a part of my MSc. in Computer Science at the University of the Witwatersrand.
We have produced a paper for this work and are currently in the submission/review process with a journal. This README will be updated with details should the work pass review for publication.

Authored by: Joshua Bruton  
Supervised by: Dr. Hairong Wang  

## Contents
This repository contains implementations or usage of the following techniques and architectures:
1. Implementations of UNet, BNet (an extension of VNet), and TSCNet
2. Translated Skip Connections, used in the TSCNet architecture (marked with comment)

We make use of Pytorch with Pytorch Lightning for our implementations.

## Usage
I have created a requirements file. I recommend using [pipenv](https://pypi.org/project/pipenv/) with Python 3.6 to open a shell and then using
~~~
pipenv install -r requirements.txt
~~~
and requirements should be met. Of course, Conda, and any other environment manager you are familiar with will work as well.

## Future work
This repository is licensed under the GNU General Public License and therefore is completely free to use for any project you see fit. If you do use or learn from our work, we would appreciate a citation, we will make the details available here after publication as this work is still in the review process.

## Suggestions
If there are any pressing problems with the code please open an issue and I will attend to it as timeously as is possible.
