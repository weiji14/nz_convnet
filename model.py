import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, AlphaDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model#, multi_gpu_model
from keras import backend as K
import matplotlib.pyplot as plt

# Design our model architecture here
def keras_model(img_width=256, img_height=256, tensorboard_images=False):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    Architecture inspired by https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
    '''
    #n_ch_exps = [4, 5, 6, 7, 8]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    n_ch_exps = [6, 6, 6, 6, 6]
    k_size = (3, 3)                     #size of filter kernel
    k_init = 'lecun_normal'             #kernel initializer
    activation = 'selu'

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (3, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)
    if tensorboard_images == True:
        tf.summary.image(name='input', tensor=inp)
    encodeds = []

    # encoder
    enc = inp
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        with K.name_scope('Encode_block_'+str(l_idx)):
            enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation=activation, padding='same', kernel_initializer=k_init)(enc)
            enc = AlphaDropout(0.1*l_idx,)(enc)
            enc = Conv2D(filters=2**n_ch, kernel_size=k_size, dilation_rate=(2,2), activation=activation, padding='same', kernel_initializer=k_init)(enc)
            encodeds.append(enc)
            #print(l_idx, enc)
            if l_idx < len(n_ch_exps)-1:  #do not run max pooling on the last encoding/downsampling step
                enc = MaxPooling2D(pool_size=(2,2))(enc)  #strides = pool_size if strides is not set
                #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation=activation, padding='same', kernel_initializer=k_init)(enc)
            if tensorboard_images == True:
                tf.summary.histogram("conv_encoder", enc)
            
    # decoder
    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        with K.name_scope('Decode_block_'+str(l_idx)):
            l_idx_rev = len(n_ch_exps) - l_idx - 1  #
            dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
            dec = Conv2D(filters=2**n_ch, kernel_size=k_size, dilation_rate=(2,2), activation=activation, padding='same', kernel_initializer=k_init)(dec)
            dec = AlphaDropout(0.1*l_idx)(dec)
            dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation=activation, padding='same', kernel_initializer=k_init)(dec)
            dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation=activation, padding='same', kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)
    if tensorboard_images == True:
        tf.summary.image(name='output', tensor=outp)
    
    model = Model(inputs=[inp], outputs=[outp])
    
    return model