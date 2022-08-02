
'''
Multiscale Residual Attention Convolutional Neural Network for Bearing Fault Diagnosis
IEEE Tr
'''
#%% import libs
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize


def cal_index(y_true, y_pred):
    '''
    Calculate the following four evaluation metrics
    Accuracy, Recall, Precision, F1-Score
    '''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro',labels=np.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='macro',labels=np.unique(y_pred))
    F1_score = f1_score(y_true, y_pred, average='macro',labels=np.unique(y_pred))

    return acc, prec, recall, F1_score

def cal_auc_roc(y_true, y_pred_score, num_class=7):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    y_true_one_hot = label_binarize(y_true, classes=list(range(num_class)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc, fpr, tpr

#%%
def cnn_1d_blk(x, filters=16, kernel_size=3, strides=1, padding='same'):
    '''
    Description: the simple encapsulation of "conv+bn+relu"
    Args:
    - x:         the input 1D-signal
    - strides:   the stride of the convolutional operation
    - padding:   'same' or 'valid'
    - filters:   the channel number of output feature maps
    - kernel_size: the kernel size of the convolutional operation
    '''
    x1 = layers.Conv1D(filters, kernel_size, strides, padding)(x)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.Activation('relu')(x2)

    return x3
#%%
def MLMod_blk(x, basic_kernel_size, scale = 6, width = 16, stride = 1, padding = 'same', multiscale=False):
    '''
    Description: MLMod
    Args:
    - x:                 the input signal
    - basic_kernel_size: the long kernal size before Multiscale block
    - scale:             the number of scale
    - width:             the channel number of each scale
    - stride:            strider of the basic convolution
    - padding:           all of conv in this work is 'same'
    '''
    # base-kernel layer
    x0 = layers.Conv1D(filters=scale*width, kernel_size=basic_kernel_size, strides=stride, padding=padding)(x)

    if multiscale:
        spx = tf.split(x0,  num_or_size_splits = scale, axis=2) # input_size is (10, 1024,96)
        for i in range(scale):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # sp = cnn_1d_blk(sp, filters=width, kernel_size=3)
            sp = layers.Conv1D(filters=width, kernel_size=5, strides=1, padding='same')(sp)

            if i==0:
                out = sp
            else:
                out = tf.concat([out, sp], axis=2)

        out = layers.BatchNormalization()(out)
        out = layers.Activation('relu')(out)
    else:
        out = layers.BatchNormalization()(x0)
        out = layers.Activation('relu')(out)
    return out



def RAMod_blk(x):
    '''
    Description: RAMod
    Args:
    - x:                 the input signal
    - basic_kernel_size: the long kernal size before Multiscale block
    - scale:             the number of scale
    - width:             the channel number of each scale
    - stride:            strider of the basic convolution
    - padding:           all of conv in this work is 'same'
    '''

    x1_C = layers.GlobalAveragePooling1D()(x) # (B, C)
    x1_C = tf.expand_dims(x1_C, axis=1)
    x1_L = layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same')(x) # (B,L)

    x2_L_C = tf.linalg.matmul(x1_L, x1_C)
    x2_L_C = layers.BatchNormalization()(x2_L_C)

    x3 = layers.Activation('sigmoid')(x2_L_C)

    x4 =  tf.math.multiply(x3, x) + x + x3 # RAL

    return x4


#%% The MRA_CNN model
def MRA_CNN(data_shape=(1024,1), basic_kernel_sizes=[32,16,16,8,4], scales=[4,4,4,4,4], widths=[16,16,32,32,64], strides=[2,2,2,2,2], class_number=10):
    lengths = np.array([len(basic_kernel_sizes), len(scales), len(widths), len(strides)])
    if all(lengths-np.mean(lengths)):
        print('Please recheck the lengths of the inputs!')

    in_x = layers.Input(shape=data_shape)
    x = in_x
    for i in range(len(basic_kernel_sizes)):
        x1 = MLMod_blk(x, basic_kernel_sizes[i], scale = scales[i], width = widths[i], stride=strides[i], multiscale=True)
        x  = RAMod_blk(x1)

    x = layers.GlobalAveragePooling1D()(x)
    out_x = layers.Dense(class_number, activation='softmax')(x)
    model = tf.keras.Model(inputs=in_x, outputs=out_x)

    return model


#%%
#%
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

def laplace_noise(x, snr, u=0, b =1):

    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr

    lp_n = np.random.laplace(u,b, len(x))
    lp_n = lp_n/np.sqrt((np.sum(lp_n**2)/len(lp_n)))

    return  lp_n * np.sqrt(npower)


def pink_noise(x, snr):
    # https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr

    X_white = np.fft.rfft(np.random.randn(len(x)))

    h = lambda t: 1/np.where(t == 0, float('inf'), np.sqrt(t))
    S = h(np.fft.rfftfreq(len(x)))
    X_shaped = X_white * S
    pink_n = np.fft.irfft(X_shaped)

    pink_n = pink_n/np.sqrt((np.sum(pink_n**2)/len(pink_n)))

    return  pink_n * np.sqrt(npower)

def add_noise(data, snr_num, noise_type = 'gaussian'):

    if noise_type == 'gaussian':
        rand_data = wgn(data, snr_num)
    elif noise_type == 'laplace':
        rand_data = laplace_noise(data, snr_num,u =0, b =1)
    elif noise_type == 'pink':
        rand_data = pink_noise (data, snr_num)

    data = data + rand_data

    return data

def calc_flops(model):
    '''
    Reference:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md
    # https://github.com/foolmarks/flops/blob/main/flops.py
    '''
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    from tensorflow.python.profiler.model_analyzer import profile
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    #flops = graph_info.total_float_ops // 2
    flops = graph_info.total_float_ops
    print('Flops: {:,}'.format(flops))

    return flops

