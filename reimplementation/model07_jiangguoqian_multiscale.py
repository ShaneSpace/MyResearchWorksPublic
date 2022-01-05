"""
Jiang G, He H, Yan J, et al. Multiscale convolutional neural networks for fault diagnosis of wind turbine gearbox[J]. IEEE Transactions on Industrial Electronics, 2018, 66(4): 3196-3207.

@article{jiang2018multiscale,
  title={Multiscale convolutional neural networks for fault diagnosis of wind turbine gearbox},
  author={Jiang, Guoqian and He, Haibo and Yan, Jun and Xie, Ping},
  journal={IEEE Transactions on Industrial Electronics},
  volume={66},
  number={4},
  pages={3196--3207},
  year={2018},
  publisher={IEEE}
}


This code is reimplemented by JIA Linshan
Github: ShaneSpace
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn.model_selection import train_test_split


def conv1d_bn_relu_maxpool(x, filters,kernel_size,padding,strides):
    xx = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    xx = BatchNormalization()(xx)
    xx = Activation('relu')(xx)
    xx = Dropout(0.5)(xx)
    xx = MaxPooling1D(pool_size=2)(xx)

    return xx


def MSC_1DCNN(class_number):
    input_signal = Input(shape=(1024,1))
    x01 = input_signal
    x02 = AveragePooling1D(pool_size=2)(input_signal)
    x03 = AveragePooling1D(pool_size=3)(input_signal)

    x1 = conv1d_bn_relu_maxpool(x01, filters=16, kernel_size=100, padding='same', strides=1)
    x1 = conv1d_bn_relu_maxpool(x1,  filters=32, kernel_size=100, padding='same', strides=1)
    x1 = Flatten()(x1)

    x2 = conv1d_bn_relu_maxpool(x02, filters=16, kernel_size=100, padding='same', strides=1)
    x2 = conv1d_bn_relu_maxpool(x2,  filters=32, kernel_size=100, padding='same', strides=1)
    x2 = Flatten()(x2)

    x3 = conv1d_bn_relu_maxpool(x03, filters=16, kernel_size=100, padding='same', strides=1)
    x3 = conv1d_bn_relu_maxpool(x3,  filters=32, kernel_size=100, padding='same', strides=1)
    x3 = Flatten()(x3)

    x4 = concatenate([x1,x2,x3], axis=-1)
    x5 = Dropout(0.5)(x4)
    x6 = Dense(1024, activation='relu')(x5)
    output = Dense(class_number, activation='softmax')(x6)

    model = Model(inputs=input_signal, outputs=output)
    return model

my_model = MSC_1DCNN(class_number=9)
my_model.summary()
