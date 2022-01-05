'''
Peng D, Wang H, Liu Z, et al. Multibranch and multiscale CNN for fault diagnosis of wheelset bearings under strong noise and variable load condition[J]. IEEE Transactions on Industrial Informatics, 2020, 16(7): 4949-4960.

@article{peng2020multibranch,
  title={Multibranch and multiscale CNN for fault diagnosis of wheelset bearings under strong noise and variable load condition},
  author={Peng, Dandan and Wang, Huan and Liu, Zhiliang and Zhang, Wei and Zuo, Ming J and Chen, Jian},
  journal={IEEE Transactions on Industrial Informatics},
  volume={16},
  number={7},
  pages={4949--4960},
  year={2020},
  publisher={IEEE}
}

This code is reimplemented by JIA Linshan
Github: ShaneSpace
'''
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import scipy.io as sio
#%%
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

#%% There are three branches, we need to first construct the data-preprocessing things.
# average moving
def moving_average(x, w=5, padding = 'same'):
    return np.convolve(x, np.ones(w), padding) / w

def guassian_func(x):
    delta = 1
    return 1/(delta*np.sqrt(2*np.pi))*np.exp(-x*x/(2*delta*delta))

def gaussian_filtering(x,padding='same'):
    w = 5
    w_j = np.arange(5)-2
    guassian_coef = [guassian_func(i) for i in w_j]

    return np.convolve(x, guassian_coef, padding)/sum(guassian_coef)

def conv1d_block(x,filters,kernel_size,stride,dr):
    x = layers.Conv1D(filters = filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    y = layers.Dropout(dr)(x)

    return y

def multiscale_module(x, h=4, K=6,C=16,S=2,D=0.5):
    kernel_sizes = [(2**i)*K for i in range(h)]
    all_scales = []
    for i in range(h):
        x0 = conv1d_block(x,int(C/h),kernel_sizes[i],stride=S, dr=D)
        all_scales.append(x0)

    y = tf.concat(all_scales, axis=2)

    return y

def conv_branch(x):
    x = conv1d_block(x,filters=16,kernel_size=6,stride=4,dr=0.5)
    x = conv1d_block(x,filters=32,kernel_size=5,stride=2,dr=0.4)
    x = conv1d_block(x,filters=64,kernel_size=4,stride=2,dr=0.3)
    x = conv1d_block(x,filters=128,kernel_size=3,stride=2,dr=0.2)
    y = conv1d_block(x,filters=256,kernel_size=2,stride=2,dr=0.1)

    return y

def multiscale_branch(x):
    x = multiscale_module(x, h=4, K=6,C=16,S=4,D=0.5)
    x = multiscale_module(x, h=4, K=5,C=32,S=2,D=0.4)
    x = multiscale_module(x, h=4, K=4,C=64,S=2,D=0.3)
    x = multiscale_module(x, h=4, K=3,C=128,S=2,D=0.2)
    y = multiscale_module(x, h=4, K=2,C=256,S=2,D=0.1)
    return y

def MBSCNN(class_number):
    raw_signal = layers.Input(shape=(1024,1))
    lof_signal = layers.Input(shape=(1024,1)) #low frquency
    den_signal = layers.Input(shape=(1024,1)) # denoising

    y1 = multiscale_branch(raw_signal)
    y2 = conv_branch(lof_signal)
    y3 = conv_branch(den_signal)

    y = tf.concat([y1, y2, y3], axis=2)
    y = layers.GlobalAvgPool1D()(y)
    out = layers.Dense(class_number, activation='softmax')(y)

    model = tf.keras.Model(inputs=[raw_signal,lof_signal,den_signal], outputs=out)


    return model


# %%
my_model = MBSCNN(class_number=7)
my_model.summary()
#%% the reference code for loading dataset

# datafile_path = 'Data/my_data'
# data = sio.loadmat(datafile_path)
# X_train = data['X_train']
# X_test = data['X_test']
# X_valid = data['X_valid']

# x_train_raw = X_train
# x_train_den = np.array([moving_average(data) for data in X_train])
# x_train_lof = np.array([gaussian_filtering(data) for data in X_train])

# x_test_raw = X_test
# x_test_den = np.array([moving_average(data) for data in X_test])
# x_test_lof = np.array([gaussian_filtering(data) for data in X_test])


# x_valid_raw = X_valid
# x_valid_den = np.array([moving_average(data) for data in X_valid])
# x_valid_lof = np.array([gaussian_filtering(data) for data in X_valid])




