#@Time      :2018/11/9 11:02
#@Author    :zhounan
# @FileName: data_process.py

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from skmultilearn.dataset import load_from_arff

def pro_iris_data():
    x = np.load('./dataset/iris/x_train.npy')
    y = np.load('./dataset/iris/y_train.npy')
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=0)
    np.save('./dataset/iris/x_train.npy', x_train)
    np.save('./dataset/iris/y_train.npy', y_train.astype('int64'))
    
    np.save('./dataset/iris/x_test.npy', x_test)
    np.save('./dataset/iris/y_test.npy', y_test.astype('int64'))


def arff2npy(dataset_name, label_count, split=True):

    if split:
        train_file_path = './dataset/original_dataset/' + dataset_name + '/' + dataset_name + '-train.arff'
        test_file_path = './dataset/original_dataset/' + dataset_name + '/' + dataset_name + '-test.arff'

        x_train, y_train = load_from_arff(train_file_path,
                                          label_count=label_count,
                                          input_feature_type='float',
                                          label_location='end',
                                          load_sparse=False)
        x_test, y_test = load_from_arff(test_file_path,
                                          label_count=label_count,
                                          input_feature_type='int',
                                          label_location='end',
                                          load_sparse=False)
    else:
        file_path = './dataset/original_dataset/' + dataset_name + '/' + dataset_name + '.arff'
        x, y = load_from_arff(file_path,
                              label_count=label_count,
                              input_feature_type='int',
                              label_location='end',
                              load_sparse=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)

    # x_train = x_train.astype(np.float64)
    # y_train = y_train.astype(np.int64)
    # x_test = x_test.astype(np.float64)
    # y_test = y_test.astype(np.int64)

    np.save('./dataset/'+ dataset_name +'/x_train.npy', x_train.toarray())
    np.save('./dataset/'+ dataset_name +'/y_train.npy', y_train.toarray())
    np.save('./dataset/'+ dataset_name +'/x_test.npy', x_test.toarray())
    np.save('./dataset/'+ dataset_name +'/y_test.npy', y_test.toarray())

def loadnpy(dataset_name):
    x_train = np.load('./dataset/'+ dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/'+ dataset_name + '/y_train.npy')

    print(x_train.shape, y_train.shape)

if __name__ == '__main__':
    dataset_name = 'yeast'
    arff2npy(dataset_name, label_count=14, split=True)
    loadnpy(dataset_name)
