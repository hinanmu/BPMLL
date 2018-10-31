#@Time      :2018/10/31 22:23
#@Author    :zhounan
# @FileName: bp_mll_test.py
import numpy as np
import tensorflow as tf
from learning_error import hamming_loss

def predict(test_x):
    with tf.Session() as sess:
        saver  = tf.train.import_meta_graph('./tf_model/model.meta')
        saver.restore(sess, 'tf_model/model.data-00000-of-00001')
    graph = tf.get_default_graph()
    y_pred = tf.get_collection('pred_network')[0]
    x = graph.get_operation_by_name('input_x').outputs[0]
    sess.run(y_pred, feed_dict={x: test_x})

    return y_pred

def load_data():
    train_x = np.load('dataset/train_x.npy')
    train_y = np.load('dataset/train_y.npy')
    test_x = np.load('dataset/test_x.npy')
    test_y = np.load('dataset/test_y.npy')

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    _, _, test_x, test_y = load_data()
    y_pred = predict(test_x)
    loss = hamming_loss(y_pred, test_y)
    print(loss)