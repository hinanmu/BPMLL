#@Time      :2018/10/30 19:03
#@Author    :zhounan
# @FileName: bp_mll_train.py
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from operator import itemgetter

def train(data_x, data_y, dataset_name):
    data_num = data_x.shape[0]
    feature_num = data_x.shape[1]
    label_num = data_y.shape[1]
    hidden_unit = int(feature_num * 0.8)
    alpha = 0.1
    batch_size = 32

    x = tf.placeholder(tf.float32, shape=[None, feature_num], name='input_x')
    y = tf.placeholder(tf.float32, shape=[None, label_num], name='input_y')

    w1 = tf.Variable(tf.random_normal([feature_num, hidden_unit], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([hidden_unit, label_num], stddev=1, seed=1))

    bias1 = tf.Variable(tf.random_normal([hidden_unit], stddev=0.01, seed=1))
    bias2 = tf.Variable(tf.random_normal([label_num], stddev=0.01, seed=1))

    a = tf.nn.tanh(tf.matmul(x, w1) + bias1)
    pred = tf.nn.tanh(tf.matmul(a, w2) + bias2)
    tf.add_to_collection('pred_network', pred)
    loss = loss_fun(y, pred) + tf.contrib.layers.l2_regularizer(alpha)(w1) + tf.contrib.layers.l2_regularizer(alpha)(w2)

    optimazer = tf.train.AdamOptimizer(0.05).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps = int(500 * data_num / batch_size)

        for i in range(steps):
            start = (i * batch_size) % data_num
            end = min(start + batch_size, data_num)

            sess.run(optimazer, feed_dict={x: data_x[start:end],
                                           y: data_y[start:end]})

        pred = sess.run(pred, feed_dict={x: data_x})
        train_threshold(data_x, data_y, pred, dataset_name)
        saver = tf.train.Saver()
        saver.save(sess, './tf_model/' + dataset_name + '/model')

def train_threshold(data_x, data_y, pred, dataset_name):
    data_num = data_x.shape[0]
    label_num = data_y.shape[1]
    threshold = np.zeros([data_num])

    for i in range(data_num):
        pred_i = pred[i, :]
        x_i = data_x[i, :]
        y_i = data_y[i, :]
        tup_list = []
        for j in range(len(pred_i)):
            tup_list.append((pred_i[j], y_i[j]))

        tup_list = sorted(tup_list, key=itemgetter(0))
        min_val = label_num
        for j in range (len(tup_list) - 1):
            val_measure = 0

            for k in range(j + 1):
                if(tup_list[k][1] == 1):
                    val_measure = val_measure + 1
            for k in range(j + 1, len(tup_list)):
                if(tup_list[k][1] == 0):
                    val_measure = val_measure + 1

            if val_measure < min_val:
                min_val = val_measure
                threshold[i] = (tup_list[j][0] + tup_list[j + 1][0]) / 2

    linreg = Ridge(alpha=0.1)
    linreg.fit(pred, threshold)
    joblib.dump(linreg, './sk_model/' + dataset_name + '/linear_model.pkl')

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

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')
    x_train, y_train = eliminate_data(x_train, y_train)

    return x_train, y_train, x_test, y_test

#eliminate some data that have full true labels or full false labels
#移除全1或者全0标签
def eliminate_data(data_x, data_y):
    data_num = data_y.shape[0]
    label_num = data_y.shape[1]
    full_true = np.ones(label_num)
    full_false = np.zeros(label_num)

    i = 0
    while(i < len(data_y)):
        if (data_y[i] == full_true).all() or (data_y[i] == full_false).all():
            data_y = np.delete(data_y, i, axis=0)
            data_x = np.delete(data_x, i, axis=0)
        else:
            i = i + 1

    return data_x, data_y

if __name__ == '__main__':
    dataset_names = ['yeast','delicious']
    dataset_name = dataset_names[0]
    x_train, y_train, _, _ = load_data(dataset_name)
    train(x_train, y_train, dataset_name)
    #train(x_train, y_train)