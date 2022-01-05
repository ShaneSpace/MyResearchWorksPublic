"""
Huang W, Cheng J, Yang Y, et al. An improved deep convolutional neural network with multi-scale information for bearing fault diagnosis[J]. Neurocomputing, 2019, 359: 77-92.

@article{huang2019improved,
  title={An improved deep convolutional neural network with multi-scale information for bearing fault diagnosis},
  author={Huang, Wenyi and Cheng, Junsheng and Yang, Yu and Guo, Gaoyuan},
  journal={Neurocomputing},
  volume={359},
  pages={77--92},
  year={2019},
  publisher={Elsevier}
}

This code is reimplemented by JIA Linshan
Github: ShaneSpace
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *



#%%
def conv1d_bn_relu_maxpool(x, filters,kernel_size,padding,strides):
    xx = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    xx = BatchNormalization()(xx)
    xx = Activation('relu')(xx)
    xx = Dropout(rate=0.4)(xx)
    xx = MaxPooling1D(pool_size=2)(xx)

    return xx

def MSC_1DCNN(class_number):
    input_signal = Input(shape=(1024,1))
    x01 = Conv1D(1, kernel_size = 100, padding='valid', strides=1)(input_signal)
    x02 = Conv1D(1, kernel_size = 200, padding='valid', strides=1)(input_signal)
    x03 = Conv1D(1, kernel_size = 300, padding='valid', strides=1)(input_signal)

    x1 = concatenate([x01,x02,x03], axis=-2)
    x2 = conv1d_bn_relu_maxpool(x1, filters=8, kernel_size = 8, padding = 'valid', strides=2)
    x3 = conv1d_bn_relu_maxpool(x2, filters=8, kernel_size = 32,padding = 'valid', strides=4)
    x4 = conv1d_bn_relu_maxpool(x3, filters=8, kernel_size = 16,padding = 'valid', strides=2)
    x5 = Flatten()(x4)
    x6 = Dense(112, activation='relu')(x5)
    output = Dense(class_number, activation='softmax')(x6)

    model = Model(inputs=input_signal, outputs=output)
    return model

my_model = MSC_1DCNN(class_number=10)
my_model.summary()
