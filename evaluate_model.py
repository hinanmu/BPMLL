#@Time      :2018/10/12 16:27
#@Author    :zhounan
# @FileName: evaluate_model.py
import numpy as np
import scipy.io as sci

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

def find(instance, label1, label2):
    index1 = []
    index2 = []
    for i in range(instance.shape[0]):
        if instance[i] == label1:
            index1.append(i)
        if instance[i] == label2:
            index2.append(i)
    return index1, index2


def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index


def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = float("inf")
    return temp, index


def findIndex(a, b):
    for i in range(len(b)):
        if a == b[i]:
            return i


def avgprec(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    aveprec = 0
    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num,))
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            # print(loc)
            summary = summary + sum(indicator[loc:class_num]) / (class_num - loc);
        aveprec = aveprec + summary / labels_size[i]
    return aveprec / test_data_num


def Coverage(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        labels_size.append(sum(test_target[i] == 1))
        index1, index2 = find(test_target[i], 1, 0)
        labels_index.append(index1)
        not_labels_index.append(index2)

    cover = 0
    for i in range(test_data_num):
        tempvalue, index = sort(outputs[i])
        temp_min = class_num + 1
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            if loc < temp_min:
                temp_min = loc
        cover = cover + (class_num - temp_min)
    return (cover / test_data_num - 1) / class_num



def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max, index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error / num


def rloss(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    rankloss = 0
    for i in range(instance_num):
        m = labels_size[i]
        n = class_num - m
        temp = 0
        for j in range(m):
            for k in range(n):
                if temp_outputs[i][labels_index[i][j]] < temp_outputs[i][not_labels_index[i][k]]:
                    temp = temp + 1
        rankloss = rankloss + temp / (m * n)

    rankloss = rankloss / instance_num
    return rankloss


def SubsetAccuracy(predict_labels, test_target):
    test_data_num = predict_labels.shape[0]
    class_num = predict_labels.shape[1]
    correct_num = 0
    for i in range(test_data_num):
        for j in range(class_num):
            if predict_labels[i][j] != test_target[i][j]:
                break
        if j == class_num - 1:
            correct_num = correct_num + 1

    return correct_num / test_data_num


def MacroAveragingAUC(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    P = []
    N = []
    labels_size = []
    not_labels_size = []
    AUC = 0
    for i in range(class_num):
        P.append([])
        N.append([])

    for i in range(test_data_num):  # 得到Pk和Nk
        for j in range(class_num):
            if test_target[i][j] == 1:
                P[j].append(i)
            else:
                N[j].append(i)

    for i in range(class_num):
        labels_size.append(len(P[i]))
        not_labels_size.append(len(N[i]))

    for i in range(class_num):
        auc = 0
        for j in range(labels_size[i]):
            for k in range(not_labels_size[i]):
                pos = outputs[P[i][j]][i]
                neg = outputs[N[i][k]][i]
                if pos > neg:
                    auc = auc + 1
        AUC = AUC + auc / (labels_size[i] * not_labels_size[i])
    return AUC / class_num


def Performance(predict_labels, test_target):
    data_num = predict_labels.shape[0]
    tempPre = np.transpose(np.copy(predict_labels))
    tempTar = np.transpose(np.copy(test_target))
    tempTar[tempTar == 0] = -1
    com = sum(tempPre == tempTar)
    tempTar[tempTar == -1] = 0
    PreLab = sum(tempPre)
    TarLab = sum(tempTar)
    I = 0
    for i in range(data_num):
        if TarLab[i] == 0:
            I += 1
        else:
            if PreLab[i] == 0:
                I += 0
            else:
                I += com[i] / PreLab[i]
    return I / data_num


def DatasetInfo(filename):
    Dict = sci.loadmat(filename)
    data = Dict['data']
    target = Dict['target']
    data_num = data.shape[0]
    dim = data.shape[1]
    if target.shape[0] != data_num:
        target = np.transpose(target)
    labellen = target.shape[1]
    attr = 'numeric'
    if np.max(data) == 1 and np.min(data) == 0:
        attr = 'nominal'
    if np.min(target) == -1:
        target[target == -1] = 0
    target = np.transpose(target)
    LCard = sum(sum(target)) / data_num
    LDen = LCard / labellen
    labellist = []
    for i in range(data_num):
        if list(target[:, i]) not in labellist:
            labellist.append(list(target[:, i]))
    LDiv = len(labellist)
    PLDiv = LDiv / data_num
    print('|S|:', data_num)
    print('dim(S):', dim)
    print('L(S):', labellen)
    print('F(S):', attr)
    print('LCard(S):', LCard)
    print('LDen(S):', LDen)
    print('LDiv(S):', LDiv)
    print('PLDiv(S):', PLDiv)


def Friedman(N, k, r):
    r2 = [r[i] ** 2 for i in range(k)]
    temp = (sum(r2) - k * ((k + 1) ** 2) / 4) * 12 * N / k / (k + 1)
    F = (N - 1) * temp / (N * (k - 1) - temp)
    return F
