import tensorflow as tf
from tensorflow.keras import layers
# https://datascience.stackexchange.com/questions/58884/how-to-create-custom-activation-functions-in-keras-tensorflow
'''
Zhao M, Zhong S, Fu X, et al. Deep residual shrinkage networks for fault diagnosis[J]. IEEE Transactions on Industrial Informatics, 2019, 16(7): 4681-4690.
'''
def bn_relu(x):
    x1 = layers.BatchNormalization()(x)
    x2 = layers.Activation('relu')(x1)

    return x2

# get_custom_objects().update({'custom_activation': Activation(custom_activation)})
# https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/math/generic.py#L428-L517
def soft_threshold(x, threshold, mode = 'cs', name=None):
    with tf.name_scope(name or 'soft_threshold'):
        x = tf.convert_to_tensor(x, name='x')
        threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
        if mode=='cs':
            threshold1 = tf.expand_dims(threshold, axis=2) # for rsbu_cs, where threshold(batch, 1), we need to expand the dim into (batch, 1, 1)
        elif mode=='cw':
            threshold1 = tf.expand_dims(threshold, axis=1)
        return tf.sign(x) * tf.maximum(tf.abs(x) - threshold1, 0.)


def RSBU_CS(x,filters, kernel_size, strides=1, padding='same'):
    x1 = bn_relu(x)
    x2 = layers.Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)(x1)
    x2 = bn_relu(x2)
    x3 = layers.Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = padding)(x2)
    x4_0 = tf.abs(x3)
    x4_1 = layers.GlobalAveragePooling1D()(x4_0)
    x4_2 = layers.Dense(units=x4_1.shape[1])(x4_1)
    x4_3 = bn_relu(x4_2)
    x4_4 = layers.Dense(units=1)(x4_3)
    x4_5 = layers.Activation('sigmoid')(x4_4) #this is the alpha
    x5 = tf.reduce_mean(x4_1, axis=1)
    x5 = tf.expand_dims(x5, axis=1)
    thre = x5*x4_5 # threshold
    x6 = soft_threshold(x3, thre, mode='cs') + layers.Conv1D(filters = filters, kernel_size = 1, strides = strides, padding = padding)(x)

    return x6

def RSBU_CW(x, filters, kernel_size, strides=1, padding = 'same'):
    x1 = bn_relu(x)
    x2 = layers.Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)(x1)
    x2 = bn_relu(x2)
    x3 = layers.Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = padding)(x2)
    x4_0 = tf.abs(x3)
    x4_1 = layers.GlobalAveragePooling1D()(x4_0)
    x4_2 = layers.Dense(units=x4_1.shape[1])(x4_1)
    x4_3 = bn_relu(x4_2)
    x4_4 = layers.Dense(units=x4_1.shape[1])(x4_3)
    x4_5 = layers.Activation('sigmoid')(x4_4) #this is the alpha
    thre = x4_5 * x4_1
    x5 = soft_threshold(x3, thre, mode='cw') + layers.Conv1D(filters = filters, kernel_size = 1, strides = strides, padding = padding)(x)

    return x5


def DRSN(class_num = 10, mode = 'cs'):
    x = layers.Input(shape=(1024,1))
    x1 = layers.Conv1D(4,3,2,padding='same')(x)
    if mode == 'cs':
        x2 = RSBU_CS(x1, 4, 3, 2)
        x2 = RSBU_CS(x2, 4, 3, 1)
        x2 = RSBU_CS(x2, 8, 3, 2)
        x2 = RSBU_CS(x2, 8, 3, 1)
        x2 = RSBU_CS(x2, 16, 3, 2)
        x2 = RSBU_CS(x2, 16, 3, 1)
    else:
        x2 = RSBU_CW(x1, 4,  3, 2)
        x2 = RSBU_CW(x2, 4,  3, 1)
        x2 = RSBU_CW(x2, 8,  3, 2)
        x2 = RSBU_CW(x2, 8,  3, 1)
        x2 = RSBU_CW(x2, 16, 3, 2)
        x2 = RSBU_CW(x2, 16, 3, 1)

    x3 = bn_relu(x2)
    x4 = layers.GlobalAveragePooling1D()(x3)
    x5 = layers.Dense(units=class_num, activation='softmax')(x4)

    the_model = tf.keras.Model(inputs=x, outputs = x5)

    return the_model







