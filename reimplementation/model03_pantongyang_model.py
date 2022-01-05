"""
Pan T, Chen J, Zhou Z, et al. A novel deep learning network via multiscale inner product with locally connected feature extraction for intelligent fault detection[J]. IEEE Transactions on Industrial Informatics, 2019, 15(9): 5119-5128.

@article{pan2019novel,
  title={A novel deep learning network via multiscale inner product with locally connected feature extraction for intelligent fault detection},
  author={Pan, Tongyang and Chen, Jinglong and Zhou, Zitong and Wang, Changlei and He, Shuilong},
  journal={IEEE Transactions on Industrial Informatics},
  volume={15},
  number={9},
  pages={5119--5128},
  year={2019},
  publisher={IEEE}
}


This code is reimplemented by JIA Linshan
Github: ShaneSpace
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical


from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn.model_selection import train_test_split

def conv1d_bn_relu_maxpool(x, filters,kernel_size,padding,strides):
    xx = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    xx = BatchNormalization()(xx)
    xx = Activation('relu')(xx)
    xx = MaxPooling1D(pool_size=2)(xx)

    return xx


def MSC_1DCNN(class_number):
    input_signal = Input(shape=(1024,1))
    x1 = Conv1D(filters=16, kernel_size=2, padding='valid', strides=2)(input_signal)
    x1 = MaxPooling1D(pool_size=256)(x1)
    x2 = Conv1D(filters=16, kernel_size=4, padding='valid', strides=4)(input_signal)
    x2 = MaxPooling1D(pool_size=128)(x2)
    x3 = Conv1D(filters=16, kernel_size=8, padding='valid', strides=8)(input_signal)
    x3 = MaxPooling1D(pool_size=64)(x3)
    x4 = Conv1D(filters=16, kernel_size=16, padding='valid', strides=16)(input_signal)
    x4 = MaxPooling1D(pool_size=32)(x4)
    x5 = Conv1D(filters=16, kernel_size=32, padding='valid', strides=32)(input_signal)
    x5 = MaxPooling1D(pool_size=16)(x5)
    x6 = Conv1D(filters=16, kernel_size=64, padding='valid', strides=64)(input_signal)
    x6 = MaxPooling1D(pool_size=8)(x6)
    x7 = Conv1D(filters=16, kernel_size=128, padding='valid', strides=128)(input_signal)
    x7 = MaxPooling1D(pool_size=4)(x7)
    x8 = Conv1D(filters=16, kernel_size=256, padding='valid', strides=256)(input_signal)
    x8 = MaxPooling1D(pool_size=2)(x8)

    xx = concatenate([x1,x2,x3,x4,x5,x6,x7,x8],axis=-2)
    xx = Flatten()(xx)
    xx = Dense(128, activation='relu')(xx)

    output = Dense(class_number, activation='softmax')(xx)

    model = Model(inputs=input_signal, outputs=output)
    return model

my_model = MSC_1DCNN(10)
my_model.summary()


