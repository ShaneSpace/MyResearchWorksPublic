import tensorflow as tf
from tensorflow.keras import layers
'''
Zhao J, Yang S, Li Q, et al. A new bearing fault diagnosis method based on signal-to-image mapping and convolutional neural network[J]. Measurement, 2021, 176: 109088.
'''
def STIM_CNN(class_number=10):
    input_signal = layers.Input(shape=(28,28,1))
    x1 = layers.Conv2D(32, (5,5), strides=1, padding='same')(input_signal)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPool2D(pool_size=(2,2))(x1)
    x1 = layers.Conv2D(64, (5,5), strides=1, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPool2D(pool_size=(2,2))(x1)
    x1 = layers.Dense(units=1024)(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Dense(units=256)(x1)
    x1 = layers.Dropout(0.5)(x1)

    output_signal = layers.Dense(class_number, activation='softmax')(x1)
    model = tf.keras.Model(inputs = input_signal, outputs = output_signal)

    return model

my_model = STIM_CNN(10)
my_model.summary()
