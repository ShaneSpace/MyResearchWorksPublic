import tensorflow as tf
from tensorflow.keras import layers

'''
Pei X, Zheng X, Wu J. Rotating Machinery Fault Diagnosis Through a Transformer Convolution Network Subjected to Transfer Learning[J]. IEEE Transactions on Instrumentation and Measurement, 2021, 70: 1-11.
'''
def mlp(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units, activation=tf.nn.gelu)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

def conv_bn_relu_maxpool_blk(x, filters = 64, kernel_size= 3,  padding = 'same', pool_size=2):
    x1 = layers.Conv1D(filters=filters, kernel_size= kernel_size, strides= 1, padding = padding)(x)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.Activation('relu')(x2)
    x4 = layers.MaxPool1D(pool_size=pool_size)(x3)

    return x4

def msa_ffn_add_norm(x, num_heads=5, projection_dim=200, dropout_rate = 0.1):
    attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)(x, x)
    x1 = layers.Add()([attention_output, x])
    x2 = layers.LayerNormalization(epsilon=1e-6)(x1)
    x3 = mlp(x2, hidden_units = projection_dim, dropout_rate = dropout_rate)
    x4 = layers.Add()([x3, x1])

    return x4

class LinearPosEmbedding(layers.Layer):
    def __init__(self, seq_len, projection_dim):
        super(LinearPosEmbedding, self).__init__()
        self.seq_len = seq_len
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=seq_len, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def TCN(input_shape = (20,51), projection_dim=200, class_num = 10):
    input_signal = layers.Input(shape=input_shape)
    x1 = LinearPosEmbedding(input_signal.shape[1], projection_dim)(input_signal)
    x2 = msa_ffn_add_norm(x1, num_heads=5, projection_dim = projection_dim, dropout_rate = 0.1)
    x3 = msa_ffn_add_norm(x2, num_heads=5, projection_dim = projection_dim, dropout_rate = 0.1)
    x3 = tf.transpose(x3, perm=[0,2,1])
    x4 = conv_bn_relu_maxpool_blk(x3, filters = 64, kernel_size= 3,  padding = 'same', pool_size=2)
    x5 = conv_bn_relu_maxpool_blk(x4, filters = 64, kernel_size= 3,  padding = 'same', pool_size=2)
    x6 = conv_bn_relu_maxpool_blk(x5, filters = 64, kernel_size= 3,  padding = 'same', pool_size=2)

    x6 = tf.transpose(x6, perm=[0,2,1])
    x7 = layers.GlobalAveragePooling1D()(x6)
    output_signal = layers.Dense(units=class_num, activation='softmax')(x7)

    the_model = tf.keras.Model(inputs = input_signal, outputs = output_signal)

    return the_model