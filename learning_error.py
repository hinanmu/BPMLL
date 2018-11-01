#@Time      :2018/10/12 16:27
#@Author    :zhounan
# @FileName: learning_error.py
import numpy as np

def hamming_loss(test_y, predict):
    test_y = test_y.astype(np.int32)
    predict = predict.astype(np.int32)
    label_num = test_y.shape[1]
    test_data_num = test_y.shape[0]
    hmloss = 0
    temp = 0

    for i in range(test_data_num):
        temp = temp + np.sum(test_y[i] ^ predict[i])
    #end for
    hmloss = temp / label_num / test_data_num

    return hmloss

def OneError(test_y, predict):
    return  0


