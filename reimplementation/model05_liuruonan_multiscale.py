"""
Liu R, Wang F, Yang B, et al. Multiscale kernel based residual convolutional neural network for motor fault diagnosis under nonstationary conditions[J]. IEEE Transactions on Industrial Informatics, 2019, 16(6): 3797-3806.

@article{liu2019multiscale,
  title={Multiscale kernel based residual convolutional neural network for motor fault diagnosis under nonstationary conditions},
  author={Liu, Ruonan and Wang, Fei and Yang, Boyuan and Qin, S Joe},
  journal={IEEE Transactions on Industrial Informatics},
  volume={16},
  number={6},
  pages={3797--3806},
  year={2019},
  publisher={IEEE}
}


This code is reimplemented by JIA Linshan
Github: ShaneSpace
"""
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import  ReduceLROnPlateau
from tensorflow.keras.models import load_model

from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn.model_selection import train_test_split
import pandas as pd


def normalization_processing(data):
    data_mean = data.mean()
    data_var = data.var()

    data = data - data_mean
    data = data / data_var

    return data

def wgn(x, snr):

    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr

    return np.random.randn(len(x)) * np.sqrt(npower)


def add_noise(data,snr_num):

    rand_data = wgn(data, snr_num)
    data = data + rand_data

    return data


def stack_conv_block(x, filters, kernel_size, strides=1):
   C1 = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
   C1 = BatchNormalization()(C1)
   C1 = Activation('relu')(C1)
   C2 = Conv1D(filters, kernel_size, strides=strides, padding='same')(C1)
   C2 = BatchNormalization()(C2)
   xx = Conv1D(filters, 1, strides=strides, padding='same')(x)
   C2 = C2 + xx
   y = Activation('relu')(C2)

   return y

def MSC_1DCNN(class_number):
    input_signal = Input(shape=(1024,1))
    x0 = Conv1D(64,kernel_size = 7, padding='same', strides=1)(input_signal)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    x0 = MaxPooling1D(pool_size=2)(x0)

    x01 = stack_conv_block(x0, 64, 3, strides=1)
    x01 = stack_conv_block(x01, 128, 3, strides=1)
    x01 = stack_conv_block(x01, 256, 3, strides=1)
    x01 = GlobalAveragePooling1D()(x01)

    x02 = stack_conv_block(x0, 64, 5, strides=1)
    x02 = stack_conv_block(x02, 128, 5, strides=1)
    x02 = stack_conv_block(x02, 256, 5, strides=1)
    x02 = GlobalAveragePooling1D()(x02)

    x03 = stack_conv_block(x0, 64, 7, strides=1)
    x03 = stack_conv_block(x03, 128, 7, strides=1)
    x03 = stack_conv_block(x03, 256, 7, strides=1)
    x03 = GlobalAveragePooling1D()(x03)

    cat_feature_vec = concatenate([x01,x02,x03])
    x4 = Dense(768, activation='relu')(cat_feature_vec)

    output = Dense(class_number, activation='softmax')(x4)

    model = Model(inputs=input_signal, outputs=output)
    return model


my_model = MSC_1DCNN(class_number=10)
my_model.summary()
