from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import numpy as np
import csv
import gc


def build_model(i_shape, o_shape):
    model=Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(i_shape,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(o_shape, activation='linear'))
    sgd = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['acc'])

    return model


def open_data():
    with open('./data/pn_data.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        l = [row for row in reader]

    tmp1 = []
    tmp2 = []
    for i in l:
        tmp2 = [float(j) for j in i]
        tmp1.append(tmp2)
    l = tmp1

    del tmp1, tmp2
    gc.collect()

    data = []
#    for i in range(len(l)):
    for i in range(200):
        data.append(l[i])
    del l
    gc.collect()

    length = []
    for i in data:
        length.append(len(i))
    max_len = max(length)
    for i in data:
        while len(i) != max_len:
            i.append(0)
        i = np.array(i)
    data = np.array(data)
    del length, max_len
    gc.collect()

    return data


def open_score():
    with open('./data/score.txt', 'r', encoding='utf-8') as f:
        scores = f.read().split('\n')
    scores = np.array(scores)
    score = np.array(scores[scores!=''])
    del scores
    gc.collect()

    tmp = [float(i) for i in score]
    score = tmp
    del tmp
    gc.collect()

    score = np.array(score)
    score = score.reshape([len(score), 1])

#   test
    score = score[0:200]

    return score


"""
def c(data):
    catharsis = []
    for d in data:
        fun_array = []
        i = 0
        while i < len(d):
            if d[i] < 0:
            elif d[i] > 0:
            else:
            i += 1
        catharsis.append(fun_array)
    catharsis = np.array(catharsis)
    return catharsis
"""


# 前の平均ネガ + 後の平均ポジ
def c2(data):
    catharsis1 = []     # 二次元配列
    for d in data:
        fun_array = []
        fun = 0.0
        n = 0.0
        p = 0.0
        i = 0
        t = 0.0
        p_counter = 0
        while i < len(d):
            if d[i] < 0:
                t += 1.0
                n += d[i]
                fun_array.append(0.0)
                i += 1
            elif d[i] > 0:
                while i+p_counter < len(d) and d[i+p_counter] > 0:
                    p += d[i+p_counter]
                    p_counter += 1
                i += p_counter
                if t == 0.0:
                    fun = 0.0
                else:
                    fun = (n/t) + (p/p_counter)
                fun_array.append(fun)
                while p_counter > 1:
                    fun_array.append(0.0)
                    p_counter += -1
                n = 0.0
                p = 0.0
                t = 0.0
                p_counter = 0
            else:
                fun_array.append(0.0)
                i += 1
        catharsis1.append(fun_array)

    del fun_array, d, fun, n, p, i, t, p_counter
    gc.collect()

    catharsis1 = np.array(catharsis1)
    return catharsis1


# カタルシスがあったら(n->pなら)1, 無ければ0
def c1(data):
    catharsis = []
    for d in data:
        fun_array = []
        i = 0
        flag = False
        while i < len(d):
            if d[i] < 0:
                fun_array.append(0.0)
                flag = True
            elif d[i] > 0 and flag:
                fun_array.append(1.0)
                flag = False
            else:
                fun_array.append(0.0)
            i += 1
        catharsis.append(fun_array)
    catharsis = np.array(catharsis)
    return catharsis


# 連続ネガ数*カタルシス直前直後のp*n
def c3(data):
    catharsis = []
    for d in data:
        fun_array = []
        i = 0
        t = 0
        pre_n = 0.0
        while i < len(d):
            if d[i] < 0:
                pre_n = d[i]
                t += 1
                fun_array.append(0.0)
            elif d[i] > 0:
                fun = t * pre_n * d[i]
                fun_array.append(fun)
                t = 0
                pre_n = 0.0
            else:
                fun_array.append(0.0)
            i += 1
        catharsis.append(fun_array)
    catharsis = np.array(catharsis)
    return catharsis


def study():
    data = open_data()
    score = open_score()

    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split

    print('pn, c1, c2, c3.\nSelect!')
    flag = True
    while flag:
        print('study_type = ', end='', flush=True)
        study_type = input()
        if study_type == 'pn':
            flag = False
        elif study_type == 'c1':
            data = c1(data)
            flag = False
#       elif
        else:
            flag = True

    X_train,X_test,Y_train,Y_test = train_test_split(data, score, test_size=0.2)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    kf = KFold(n_splits=10, shuffle=True)
    all_loss = []
    all_val_loss = []
    all_acc = []
    all_val_acc = []
    ep = 300

    for train_index, val_index in kf.split(X_train,Y_train):
        train_data = []
        train_label = []
        val_data = []
        val_label = []
        for t in train_index:
            train_data.append(X_train[t])
            train_label.append(Y_train[t])
        for v in val_index:
            val_data.append(X_train[v])
            val_label.append(Y_train[v])
        train_data  = np.array(train_data)
        train_label = np.array(train_label).reshape(len(train_label), 1)
        val_data   = np.array(val_data)
        val_label   = np.array(val_label).reshape(len(val_label), 1)

        model = build_model(i_shape=len(train_data[0]), o_shape=len(train_label[0]))
        history = model.fit(x = train_data,
                            y = train_label,
                            epochs = ep,
                            batch_size = 8,
                            verbose = 0,
                            validation_data = (val_data, val_label))

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        all_loss.append(loss)
        all_val_loss.append(val_loss)
        all_acc.append(acc)
        all_val_acc.append(val_acc)

        test_score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test loss:', test_score[0])
        print('Test accuracy:', test_score[1])

        print('\n')

    del train_data, train_label, val_data, val_label
    del model, history, test_score
    del loss, val_loss, acc, val_acc
    del train_index, val_index
    del t, v
    gc.collect()


    ave_all_loss = np.array([np.mean([x[i] for x in all_loss]) for i in range(ep)])
    ave_all_val_loss = np.array([np.mean([x[i] for x in all_val_loss]) for i in range(ep)])
    ave_all_acc = np.array([np.mean([x[i] for x in all_acc]) for i in range(ep)])
    ave_all_val_acc = np.array([np.mean([x[i] for x in all_val_acc]) for i in range(ep)])
    del all_loss, all_val_loss, all_acc, all_val_acc
    gc.collect()

    print('loss = ', np.mean(ave_all_loss))
    print('val_loss = ', np.mean(ave_all_val_loss))
    print('acc = ', np.mean(ave_all_acc))
    print('val_acc = ', np.mean(ave_all_val_acc))

    kata1 = {'loss':np.mean(ave_all_loss),
             'val_loss':np.mean(ave_all_val_loss),
             'acc':np.mean(ave_all_acc),
             'val_acc':np.mean(ave_all_val_acc)}
    del ave_all_loss, ave_all_val_loss, ave_all_acc, ave_all_val_acc
    gc.collect()

    print(kata1)
