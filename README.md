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
    python -m ipykernel install --user  #to install conda env properly
    jupyter kernelspec list --json      #see if kernel is installed
    jupyter notebook

## Where the data goes

|Folder                 | Example of a file inside                         | extension | Notes          |
|:--------------------- |:------------------------------------------------ |:---------:|:-------------- |
| data/vector           | nz-building-outlines-pilot.shp                   | \*.shp    | see section Training data/Mask |
| data/raster/downloads | lds-tile-2015-bk39-5000-0401-rgb-GTiff.zip       | \*.zip    | see section Training data/Images  |
| data/raster           | 2015_BK39_5000_0401_RGB.tif                      | \*.tif    | unzipped files from data/raster/downloads |
| data/train            | X_2015_BK39_5000_0401_RGB.hdf5                   | \*.hdf5   | binary of tif file to load into numpy array |
| data/test             | wellington-03m-rural-aerial-photos-2012-2013.tif | \*.tif    | unzipped files similar to those in data/raster |

# Data sources used to train the [keras](https://github.com/keras-team/keras) model

Using freely available data from [LINZ Data Service](https://data.linz.govt.nz/). As there is a 3.5GB limit, we resort to using region crops using the 'Set a crop' tool on the top right. Not ideal but it ensures a little bit of reproducibility.

## Training data

### Images

|Region Crop Type                       |Region Name                          |LINZ Data Source|
| ------------------------------------- |:-----------------------------------:| --------------:|
| General Electorate Boundaries 2014    | Wigram                              | [Canterbury 0.3m Rural Aerial Photos (2015-16)](https://data.linz.govt.nz/layer/53519-canterbury-03m-rural-aerial-photos-2015-16/) |
| Manual Tile Selection\*               | Hastings 2015_BK39_5000_{XXXX}_RGB  | [0401](https://data.linz.govt.nz/x/vnGVkg) [0402](https://data.linz.govt.nz/x/aA5XSv) [0403](https://data.linz.govt.nz/x/DYsY9B) [0404](https://data.linz.govt.nz/x/qvgapR) [0405](https://data.linz.govt.nz/x/VKVcWf) [0501](https://data.linz.govt.nz/x/8hJeCu) [0502](https://data.linz.govt.nz/x/k57ftA) [0503](https://data.linz.govt.nz/x/QTuhaQ) [0504](https://data.linz.govt.nz/x/3qijGe) [0505](https://data.linz.govt.nz/x/gEXkwt) [0601](https://data.linz.govt.nz/x/KcLnd9) [0602](https://data.linz.govt.nz/x/wy9pLP) [0603](https://data.linz.govt.nz/x/bNwq2d) [0604](https://data.linz.govt.nz/x/Ekkshs) [0605](https://data.linz.govt.nz/x/r9ZuP8) |


* Manual tile selection selects tiles manually from the tiles table, e.g. [here](https://data.linz.govt.nz/layer/53401-hawkes-bay-03m-rural-aerial-photos-2014-15/data/)

### Mask

- [NZ Building Outlines (Pilot)](https://data.linz.govt.nz/layer/53413-nz-building-outlines-pilot/) in shapefile format.

## Test data

|Region Crop Type                       |Region Name             |LINZ Data Source|
| ------------------------------------- |:----------------------:| --------------:|
| NZ Topo 50 Map Sheets                 | BP31 - Porirua         | [Wellington 0.3m Rural Aerial Photos (2012-2013)](https://data.linz.govt.nz/layer/51870-wellington-03m-rural-aerial-photos-2012-2013/)


## Output examples

Sample outputs on cross validation dataset. Left is input RGB image, Middle is ConvNet model output, Right is the Mask.

![sample1](https://user-images.githubusercontent.com/23487320/36362177-17747d88-1597-11e8-8c17-167b8037cb71.png)
![sample2](https://user-images.githubusercontent.com/23487320/36362245-9dd6fa04-1597-11e8-959b-87ed3217e131.png)
![sample3](https://user-images.githubusercontent.com/23487320/36362261-bfc48046-1597-11e8-81c9-c4139569cde0.png)
