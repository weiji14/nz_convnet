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

# Data sources used

Using freely available data from [LINZ Data Service](https://data.linz.govt.nz/). As there is a 3.5GB limit, we resort to using region crops using the 'Set a crop' tool on the top right. Not ideal but it ensures a little bit of reproducibility.

## Training data

### Images

|Region Crop Type                       |Region Name             |LINZ Data Source|
| ------------------------------------- |:----------------------:| --------------:|
| General Electorate Boundaries 2014    | Wigram                 | [Canterbury 0.3m Rural Aerial Photos (2015-16)](https://data.linz.govt.nz/layer/53519-canterbury-03m-rural-aerial-photos-2015-16/) |

### Mask

- [NZ Building Outlines (Pilot)](https://data.linz.govt.nz/layer/53413-nz-building-outlines-pilot/)

## Test data

|Region Crop Type                       |Region Name             |LINZ Data Source|
| ------------------------------------- |:----------------------:| --------------:|
| NZ Topo 50 Map Sheets                 | BP31 - Porirua         | [Wellington 0.3m Rural Aerial Photos (2012-2013)](https://data.linz.govt.nz/layer/51870-wellington-03m-rural-aerial-photos-2012-2013/)


## Output examples

Sample outputs on cross validation dataset. Left is input RGB image, Middle is ConvNet model output, Right is the Mask.

![sample1](https://user-images.githubusercontent.com/23487320/36362177-17747d88-1597-11e8-8c17-167b8037cb71.png)
![sample2](https://user-images.githubusercontent.com/23487320/36362245-9dd6fa04-1597-11e8-959b-87ed3217e131.png)
![sample3](https://user-images.githubusercontent.com/23487320/36362261-bfc48046-1597-11e8-81c9-c4139569cde0.png)
