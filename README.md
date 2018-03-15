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

## Prediction

### In near realtime!

    source activate nz_convnet
    python predict.py
    
    #You can also set 2 integer parameters:
    #  1st argument - output pixel size e.g. 256, 512, 1024 (default: 256)
    #  2nd argument - prediction threshold e.g from least accurate 0% accept anything to 100% won't output much (default: 50)
    python predict.py 512 50 

Live testing on imagery of Karori, Wellington.

![livesample1](https://user-images.githubusercontent.com/23487320/36468063-aed6c1bc-1746-11e8-8337-51a6a62ec796.gif)

### To a raster geotiff (which you can vectorize to a polygon)

Once you clone the repository, open the [jupyter notebook](nz_convnet.ipynb) and follow the instructions to run 'Part 5 - Save Results'.
You will need to have some geotiffs inside the data/test folder, and you may need to tweak the (img_height, img_width) parameter.

There might need to be some fiddling on your part to get this parameter right, so that the input RGB image will be tiled perfectly.
The algorithm will create a prediction on each tile, and join in back together, so if it is not tiled perfectly due to the user setting, it will raise an error.

Below is a visualization in [QGIS 3.0](https://qgis.org/) of a sample test image and the predicted raster mask output

![qgissample1](https://user-images.githubusercontent.com/23487320/37496053-2e7a79ce-2915-11e8-9732-fd27592ba237.gif)

Mask was styled using singleband pseudocolor, linear interpolation, with the OrRd color ramp in equal-interval mode.
Opacity set to 0% for values 0, 0.25 and 0.5, and 50% for values 0.75 and 1.0.
GIF was recorded using [Peek](https://github.com/phw/peek)

### More output examples

Sample outputs on cross validation dataset plotted with matplotlib inside the [jupyter notebook](nz_convnet.ipynb) environment.
Left is input RGB image, Middle is ConvNet model output, Right is the Mask.

![sample1](https://user-images.githubusercontent.com/23487320/36362177-17747d88-1597-11e8-8c17-167b8037cb71.png)
![sample2](https://user-images.githubusercontent.com/23487320/36362245-9dd6fa04-1597-11e8-959b-87ed3217e131.png)
![sample3](https://user-images.githubusercontent.com/23487320/36362261-bfc48046-1597-11e8-81c9-c4139569cde0.png)



# Data sources used to train the [keras](https://github.com/keras-team/keras) model

Using freely available data from [LINZ Data Service](https://data.linz.govt.nz/). As there is a 3.5GB limit, we resort to using region crops using the 'Set a crop' tool on the top right. Not ideal but it ensures a little bit of reproducibility.

## Training data

### Images

|Region Crop Type                       |Region Name                          |LINZ Data Source|
| ------------------------------------- |:-----------------------------------:| --------------:|
| General Electorate Boundaries 2014    | Wigram                              | [Canterbury 0.3m Rural Aerial Photos (2015-16)](https://data.linz.govt.nz/layer/53519-canterbury-03m-rural-aerial-photos-2015-16/) |
| Manual Tile Selection\*               | Hastings 2015_BK39_5000_{XXXX}_RGB  | [0401](https://data.linz.govt.nz/x/vnGVkg) [0402](https://data.linz.govt.nz/x/aA5XSv) [0403](https://data.linz.govt.nz/x/DYsY9B) [0404](https://data.linz.govt.nz/x/qvgapR) [0405](https://data.linz.govt.nz/x/VKVcWf) [0501](https://data.linz.govt.nz/x/8hJeCu) [0502](https://data.linz.govt.nz/x/k57ftA) [0503](https://data.linz.govt.nz/x/QTuhaQ) [0504](https://data.linz.govt.nz/x/3qijGe) [0505](https://data.linz.govt.nz/x/gEXkwt) [0601](https://data.linz.govt.nz/x/KcLnd9) [0602](https://data.linz.govt.nz/x/wy9pLP) [0603](https://data.linz.govt.nz/x/bNwq2d) [0604](https://data.linz.govt.nz/x/Ekkshs) [0605](https://data.linz.govt.nz/x/r9ZuP8) |
| Manual Tile Selection\*               | Tuakau bb32_{XXXX}                  | [4630](https://data.linz.govt.nz/x/9s9M9A) [4631](https://data.linz.govt.nz/x/nGwPpQ) [4632](https://data.linz.govt.nz/x/RekRWe) [4633](https://data.linz.govt.nz/x/43ZTCt) [4634](https://data.linz.govt.nz/x/hRNUs9) [4635](https://data.linz.govt.nz/x/LoBWaP) [4636](https://data.linz.govt.nz/x/yByYGd) [4637](https://data.linz.govt.nz/x/Fwbbd8) [4638](https://data.linz.govt.nz/x/tLQdLN) [4639](https://data.linz.govt.nz/x/XiDe2c) [4730](https://data.linz.govt.nz/x/oUpiP7) [4731](https://data.linz.govt.nz/x/Srdj6M) [4732](https://data.linz.govt.nz/x/6FSmmb) [4733](https://data.linz.govt.nz/x/yAvvdB) [4734](https://data.linz.govt.nz/x/idFoTq) [4735](https://data.linz.govt.nz/x/Mz4p96) [4736](https://data.linz.govt.nz/x/zPrrqL) [4737](https://data.linz.govt.nz/x/dmftXa) [4738](https://data.linz.govt.nz/x/HAUvDp) [4739](https://data.linz.govt.nz/x/uYHwt5) [4830](https://data.linz.govt.nz/x/pgh3xo) [4831](https://data.linz.govt.nz/x/T5W5e4) [4832](https://data.linz.govt.nz/x/7TK7MJ) [4833](https://data.linz.govt.nz/x/jp883Y) [4834](https://data.linz.govt.nz/x/PDwAin) [4835](https://data.linz.govt.nz/x/2bkCQ3) [4836](https://data.linz.govt.nz/x/eyZD7H) [4837](https://data.linz.govt.nz/x/JNNFnX) [4838](https://data.linz.govt.nz/x/vkBHUm) [4839](https://data.linz.govt.nz/x/Z8yKA2) [4930](https://data.linz.govt.nz/x/DWnLrG) [4931](https://data.linz.govt.nz/x/qtbNYW) [4932](https://data.linz.govt.nz/x/VHQQEk) [4933](https://data.linz.govt.nz/x/8fDRuz) [4934](https://data.linz.govt.nz/x/k32TcF) [4935](https://data.linz.govt.nz/x/QRpVJV) [4936](https://data.linz.govt.nz/x/3odWyj) [4937](https://data.linz.govt.nz/x/gCSYfy) [4938](https://data.linz.govt.nz/x/KaFaNE) [4939](https://data.linz.govt.nz/x/ww4b4U) [5030](https://data.linz.govt.nz/x/bLrdji) [5031](https://data.linz.govt.nz/x/EiffRx) [5032](https://data.linz.govt.nz/x/r7Ug8D) [5033](https://data.linz.govt.nz/x/WVHioT) [5034](https://data.linz.govt.nz/x/9r6kVh) [5035](https://data.linz.govt.nz/x/nFtnBw) [5036](https://data.linz.govt.nz/x/RdhosC) [5037](https://data.linz.govt.nz/x/42WqZS) [5038](https://data.linz.govt.nz/x/hQKsFg) [5039](https://data.linz.govt.nz/x/Lm8tvv) 

* Manual tile selection selects tiles manually from the tiles table, e.g. [here](https://data.linz.govt.nz/layer/53401-hawkes-bay-03m-rural-aerial-photos-2014-15/data/)

### Mask

- [NZ Building Outlines (Pilot)](https://data.linz.govt.nz/layer/53413-nz-building-outlines-pilot/) in shapefile format.

## Test data

|Region Crop Type                       |Region Name             |LINZ Data Source|
| ------------------------------------- |:----------------------:| --------------:|
| NZ Topo 50 Map Sheets                 | BP31 - Porirua         | [Wellington 0.3m Rural Aerial Photos (2012-2013)](https://data.linz.govt.nz/layer/51870-wellington-03m-rural-aerial-photos-2012-2013/)
