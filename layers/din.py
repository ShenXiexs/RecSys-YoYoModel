from tensorflow import Tensor
import tensorflow as tf


def dnn():
    layers = []
    units = [64,32,1]
    for i, unit in enumerate(units):
        name = 'dnn_hidden_%s_%d' % ('din', i)
        layer = tf.keras.layers.Dense(
            units=unit, activation='relu',
            kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
            name=name
        )
        layers.append(layer)
        # if i < max(len(units) - 2, 0):
        #    dropout = tf.keras.layers.Dropout(0.2)
        #    layers.append(dropout)
    return tf.keras.Sequential(layers)


def din_layer(x_hist:Tensor,x_candi):
    '''

    :param x_hist: 历史items shape:(B,seq_max_len,emb_dim)
    :param x_candi: 当前item shape:(B,emb_dim)
    :return:
    '''
    seq_max_len = tf.shape(x_hist)[1]
    emb_dim = x_hist.shape[2]

    # print(f'seq_max_len:{seq_max_len},emb_dim:{emb_dim}')


    cur_ids = tf.tile(x_candi, [1, seq_max_len])
    # print(f'cur_ids1:{tf.shape(cur_ids)}', cur_ids)
    cur_ids = tf.reshape(cur_ids,
                         tf.shape(x_hist))  # (B, seq_max_len, emb_dim)
    # print(f'cur_ids2:{tf.shape(cur_ids)}',cur_ids)

    din_net = tf.concat(
        [cur_ids, x_hist, cur_ids - x_hist, cur_ids * x_hist],
        axis=-1)
    # print(f'din_net:{tf.shape(din_net)}')

    din_layer = dnn()

    w = din_layer(din_net) # [batch_size, seq_len, 1]
    # print(f'w1:{tf.shape(w)}',w)
    # w = tf.nn.softmax(w,axis= 1)  # , aiming to reserve the intensity of user interests That is, normalization with softmax on the output of a(·) is abandoned
    # print(f'w2:{w}')

    x_hist = tf.reduce_sum(w*x_hist, axis=1)  # [batch_size, dim]
    # print(f'x_hist---:{tf.shape(x_hist)}')
    # print(f'cur_ids---:{tf.shape(x_hist)}')
    return x_hist #tf.concat([x_hist, x_candi], axis=-1) # [batch_size,dim]


if __name__ == '__main__':
    batch_size = 5
    seq_length = 3
    embedding_dim = 2
    x_hist = tf.random.normal((batch_size, embedding_dim)) #tf.random.normal((batch_size, seq_length, embedding_dim))  # (32, 10, 128)
    x_hist = tf.expand_dims(x_hist, 1)
    print('x_hist：',x_hist)

    x_curr = tf.random.normal((batch_size, embedding_dim))
    din_out = din_layer(x_hist, x_curr)
    print(din_out)
