"""
Zhang W, Peng G, Li C, et al. A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals[J]. Sensors, 2017, 17(2): 425.

@article{zhang2017new,
  title={A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals},
  author={Zhang, Wei and Peng, Gaoliang and Li, Chuanhao and Chen, Yuanhang and Zhang, Zhujun},
  journal={Sensors},
  volume={17},
  number={2},
  pages={425},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}

This code is reimplemented by JIA Linshan
Github: ShaneSpace
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import  ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


#%%
def conv1d_bn_relu_maxpool(x, filters,kernel_size,padding,strides):
    xx = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    xx= BatchNormalization()(xx)
    xx = Activation('relu')(xx)
    xx = MaxPooling1D(pool_size=2)(xx)

    return xx

def MSC_1DCNN(class_number):
    input_signal = Input(shape=(1024,1))
    x = conv1d_bn_relu_maxpool(input_signal, 16,64,'valid',8)
    x = conv1d_bn_relu_maxpool(x, 32,3,'valid',1)
    x = conv1d_bn_relu_maxpool(x, 64,3,'valid',1)
    x = conv1d_bn_relu_maxpool(x, 64,3,'valid',1)
    x = conv1d_bn_relu_maxpool(x, 64,3,'valid',1)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(class_number, activation='softmax')(x)

    model = Model(inputs=input_signal, outputs=output)
    return model

my_model = MSC_1DCNN(class_number=10)
my_model.summary()

