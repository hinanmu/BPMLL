#@Time      :2018/10/31 22:23
#@Author    :zhounan
# @FileName: bp_mll_test.py
import numpy as np
import tensorflow as tf
from learning_error import hamming_loss
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

def predict(test_x):
    with tf.Session() as sess:
        saver  = tf.train.import_meta_graph('./tf_model/model.meta')
        saver.restore(sess, 'tf_model/model')
        graph = tf.get_default_graph()
        pred = tf.get_collection('pred_network')[0]
        x = graph.get_operation_by_name('input_x').outputs[0]

        pred = sess.run(pred, feed_dict={x: test_x})

    linreg = joblib.load('./sk_model/linear_model.pkl')
    threshold = linreg.predict(pred)
    y_pred = ((pred.T - threshold.T) > 0).T

    #translate bool to int
    y_pred = y_pred + 0
    return pred

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

def load_data():
    train_x = np.load('dataset/train_x.npy')
    train_y = np.load('dataset/train_y.npy')
    test_x = np.load('dataset/test_x.npy')
    test_y = np.load('dataset/test_y.npy')
    test_x, test_y = eliminate_data(test_x, test_y)

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    _, _, test_x, test_y = load_data()
    pred = predict(test_x)
    loss = hamming_loss(pred, test_y)
    print(loss)