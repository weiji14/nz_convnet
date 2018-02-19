Convolutional Neural Network (ConvNet) trained on open data from Land Information New Zealand (LINZ). Specifically, we train the ConvNet on aerial photography to detect building outlines.

# Getting started

## Quickstart

Launch Binder

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/nz_convnet/master)

## Installation

    git clone https://github.com/weiji14/nz_convnet.git
    cd nz_convnet
    conda env create -f environment.yml

## Running the jupyter notebook

    source activate nz_convnet
    jupyter notebook

# Data sources used:

- Training mask: https://data.linz.govt.nz/layer/53413-nz-building-outlines-pilot/
- Training data: https://data.linz.govt.nz/layer/53519-canterbury-03m-rural-aerial-photos-2015-16/
- Test data (TODO): https://data.linz.govt.nz/layer/51870-wellington-03m-rural-aerial-photos-2012-2013/
