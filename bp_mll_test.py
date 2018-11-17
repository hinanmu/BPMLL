#@Time      :2018/10/31 22:23
#@Author    :zhounan
# @FileName: bp_mll_test.py
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
import evaluate_model

def predict(x_test, dataset_name):
    with tf.Session() as sess:
        saver  = tf.train.import_meta_graph('./tf_model/' + dataset_name + '/model.meta')
        saver.restore(sess, './tf_model/' + dataset_name + '/model')
        graph = tf.get_default_graph()
        pred = tf.get_collection('pred_network')[0]
        x = graph.get_operation_by_name('input_x').outputs[0]

        pred = sess.run(pred, feed_dict={x: x_test})

    linreg = joblib.load('./sk_model/' + dataset_name + '/linear_model.pkl')
    threshold = linreg.predict(pred)
    y_pred = ((pred.T - threshold.T) > 0).T

    #translate bool to int
    y_pred = y_pred + 0
    return y_pred, pred

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

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')
    x_test, y_test = eliminate_data(x_test, y_test)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['yeast','delicious']
    dataset_name = dataset_names[0]
    _, _, x_test, y_test = load_data(dataset_name)
    pred, output = predict(x_test, dataset_name)
    print(dataset_name, 'hammingloss:',evaluate_model.hamming_loss(pred, y_test))
    print(dataset_name, 'rankingloss:',evaluate_model.rloss(output, y_test))
    print(dataset_name, 'oneerror:',evaluate_model.OneError(output, y_test))