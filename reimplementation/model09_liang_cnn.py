import tensorflow as tf
from tensorflow.keras import layers
'''
Liang H, Zhao X. Rolling bearing fault diagnosis based on one-dimensional dilated convolution network with residual connection[J]. IEEE Access, 2021, 9: 31078-31091.
'''
def RCB(x):
    '''
    residual connection block
    '''
    weight_coef = 0.2
    # input_signal = layers.Input(shape = (784,1))
    x1 = layers.Conv1D(filters=16, kernel_size=64, strides=8, padding='same')(x)#(input_signal)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.MaxPool1D(pool_size = 2)(x1)
    x3 = layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same')(x2)
    x3_r = layers.Conv1D(filters=32, kernel_size=1, strides=16, padding='same')(x)#(input_signal)
    x4 = x3+x3_r*weight_coef


    x5 = layers.BatchNormalization()(x4)
    x5 = layers.ReLU()(x5)
    output_signal = layers.MaxPool1D(pool_size = 2, padding='same')(x5)

    # the_model = tf.keras.Model(inputs = input_signal, outputs=output_signal)

    return output_signal # the_model

def DRCB(x):
    '''
    Dilated rresidual connection block
    '''
    weight_coef = 0.2
    # input_signal = layers.Input(shape = (25,32))
    x1 = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same',dilation_rate=1)(x) #(input_signal)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same',dilation_rate=2)(x1)


    x2 = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same',dilation_rate=3)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2_r = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(x) #(input_signal)

    x3 = x2 + x2_r*weight_coef

    output_signal = layers.MaxPool1D(pool_size = 2, padding='same')(x3)

    # the_model = tf.keras.Model(inputs = input_signal, outputs = output_signal)

    return output_signal #the_model

def SE_blk(x):

    x1 = layers.GlobalAveragePooling1D()(x)
    x2 = layers.Dense(units = int(x1.shape[1]//2))(x1)
    x3 = layers.Dense(units = int(x1.shape[1]))(x2)
    x4 = tf.expand_dims(x3, axis=1)

    output_signal = layers.Multiply()([x4,x])

    return output_signal

def DCNNRC(class_number):
    '''
    one-dimensional dilated convolutional neural network with residual connection method
    '''
    weight_coef = 0.2 #the lambda parameter
    x = layers.Input(shape = (784,1))
    x1 = RCB(x)
    x2 = SE_blk(x1)
    x3 = DRCB(x2)
    x3_r = layers.Conv1D(filters=64, kernel_size=1, strides=64, padding='same')(x)
    x4 = x3_r + x3*weight_coef
    x5 = SE_blk(x4)
    x6 = layers.Dense(units=100)(x5)
    x7 = layers.Dense(units=class_number, activation='softmax')(x6)

    the_model = tf.keras.Model(inputs = x, outputs=x7)

    return the_model

class_number = 10
my_model = DCNNRC(class_number)
my_model.summary()






