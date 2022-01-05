#%%
'''
Wen L, Li X, Gao L, et al. A new convolutional neural network-based data-driven fault diagnosis method[J]. IEEE Transactions on Industrial Electronics, 2017, 65(7): 5990-5998.

@article{wen2017new,
  title={A new convolutional neural network-based data-driven fault diagnosis method},
  author={Wen, Long and Li, Xinyu and Gao, Liang and Zhang, Yuyan},
  journal={IEEE Transactions on Industrial Electronics},
  volume={65},
  number={7},
  pages={5990--5998},
  year={2017},
  publisher={IEEE}
}

This code is reimplemented by JIA Linshan
Github: ShaneSpace
'''
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

#%%
#%% we will use the raw signal and gram feature representation as the input of the 2D-CNN model
def TraditonalCNNModel(class_number):
    input_signal = Input(shape=(32,32,1))
    x0 = Conv2D(filters=32, kernel_size=5, padding='same',strides=1,activation='relu')(input_signal)
    # x0 = BatchNormalization()(x0)
    x0 = MaxPooling2D(pool_size=(2,2))(x0)
    x0 = Dropout(rate=0.2)(x0)

    x1 = Conv2D(filters=64, kernel_size=3, padding='same',strides=1,activation='relu')(x0)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)
    x1 = Dropout(rate=0.2)(x1)

    x2 = Conv2D(filters=128, kernel_size=3, padding='same',strides=1,activation='relu')(x1)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2,2))(x2)
    x2 = Dropout(rate=0.2)(x2)

    x3 = Conv2D(filters=256, kernel_size=3, padding='same',strides=1,activation='relu')(x2)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2,2))(x3)
    x3 = Dropout(rate=0.2)(x3)
    x4 = GlobalAveragePooling2D()(x3)

    # x4 = Flatten()(x3)
    x5 = Dense(units = 2560, activation='relu')(x4)
    x5 = Dense(units = 768, activation='relu')(x4)
    output = Dense(class_number, activation='softmax')(x5)
    model = Model(inputs=input_signal, outputs=output)
    return model


my_model = TraditonalCNNModel(class_number=10)
my_model.summary()