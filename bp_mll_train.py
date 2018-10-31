#@Time      :2018/10/30 19:03
#@Author    :zhounan
# @FileName: bp_mll_train.py
import numpy as np
import tensorflow as tf

def train(data_x, data_y):
    batch_size = 64
    data_num = data_x.shape[0]
    feature_num = data_x.shape[1]
    label_num = data_y.shape[1]
    x = tf.placeholder(tf.float32, shape=[None, feature_num])
    y = tf.placeholder(tf.float32, shape=[None, label_num])

    w1 = tf.Variable(tf.random_normal([feature_num, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, label_num], stddev=1, seed=1))

    bias1 = tf.Variable(tf.random_normal([3], stddev=1, seed=1))
    bias2 = tf.Variable(tf.random_normal([label_num], stddev=1, seed=1))

    a = tf.nn.tanh(tf.matmul(x, w1) + bias1)
    y_pre = tf.nn.tanh(tf.matmul(a, w2) + bias2)
    loss = loss_fun(y, y_pre)

    optimazer = tf.train.AdamOptimizer(0.001).minimize(loss)

    print('begin to train')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps = 5000

        for i in range(steps):
            start = (i * batch_size) % data_num
            end = min(start + batch_size, data_num)
            sess.run(optimazer, feed_dict={x: data_x[start:end],
                                           y: data_y[start:end]})

        saver = tf.train.Saver()
        saver.save(sess, "./tf_model/model")


def loss_fun(y, y_pre):
    shape = tf.shape(y)
    y_i = tf.equal(y, tf.ones(shape))
    y_not_i = tf.equal(y, tf.zeros(shape))

    # get indices to check
    truth_matrix = tf.to_float(pairwise_and(y_i, y_not_i))

    # calculate all exp'd differences
    # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
    sub_matrix = pairwise_sub(y_pre, y_pre)
    exp_matrix = tf.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1,2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_not_i), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    loss = tf.divide(sums, normalizers)

    return loss

# compute pairwise differences between elements of the tensors a and b
def pairwise_sub(a, b):
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.subtract(column, row)

# compute pairwise logical and between elements of the tensors a and b
#if y shape is [3,3], y_i would be translate to [3,3,1], y_not_i is would be [3,1,3]
#and return [3,3,3],through the matrix ,we can easy to caculate c_k - c_i(appear in the paper)
def pairwise_and(a, b):
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.logical_and(column, row)

def load_data():
    train_x = np.load('dataset/train_x.npy')
    train_y = np.load('dataset/train_y.npy')
    test_x = np.load('dataset/test_x.npy')
    test_y = np.load('dataset/test_y.npy')

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    train_y
    train(train_x, train_y)