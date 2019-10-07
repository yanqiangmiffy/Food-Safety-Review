#! -*- coding:utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
import re
import jieba
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
import sys
import ipykernel

pd.set_option('display.max_columns', None)

learning_rate = 5e-5
min_learning_rate = 1e-5

base = 'D:/data/bert/chinese_wwm_ext_L-12_H-768_A-12/'
config_path = base + 'bert_config.json'
checkpoint_path = base + 'bert_model.ckpt'
dict_path = base + 'vocab.txt'
# MAX_LEN = 30
MAX_LEN = 100

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

train = pd.read_csv('data/train.csv', sep='\t')
test = pd.read_csv('data/test_new.csv')
sub = pd.read_csv('data/sample.csv')

train_comment = train['comment'].values
test_comment = test['comment'].values

labels = train['label'].astype(int).values


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=8):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X, y = self.data

            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                d = X[i]
                text = d[:MAX_LEN]
                t, t_ = tokenizer.encode(first=text)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    Y = np.array(Y)
                    T = seq_padding(T)
                    T_ = seq_padding(T_)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(1, activation='sigmoid')(T)

    model = Model([T1, T2], output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('./models/bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (
            self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_y = self.val_data
        for i in tqdm(range(len(val_x1))):
            d = val_x1[i]
            text = d[:MAX_LEN]

            t1, t1_ = tokenizer.encode(first=text)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0] + 1)
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y + 1, self.predict))
        acc = accuracy_score(val_y + 1, self.predict)
        f1 = f1_score(val_y + 1, self.predict, average='macro')
        return score, acc, f1


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def predict(data):
    prob = []
    val_x1 = data
    for i in tqdm(range(len(val_x1))):
        X = val_x1[i]
        text = X[:MAX_LEN]
        t1, t1_ = tokenizer.encode(first=text)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob


oof_train = np.zeros((len(train), 1), dtype=np.float32)
oof_test = np.zeros((len(test), 1), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_comment, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_comment[train_index]
    y = labels[train_index]

    val_x1 = train_comment[valid_index]
    val_y = labels[valid_index]

    train_D = data_generator([x1, y])
    evaluator = Evaluate([val_x1, val_y], valid_index)

    model = get_model()
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=4,
                        callbacks=[evaluator]
                        )
    model.load_weights('./models/bert{}.w'.format(fold))
    oof_test += predict(test_comment)
    K.clear_session()
    test['flag'] = oof_test

oof_test /= 4

np.savetxt('./models/train_bert.txt', oof_train)
np.savetxt('./models/test_bert.txt', oof_test)

oof_train = oof_train.reshape(-1)
cv_score = roc_auc_score(labels, oof_train)
print(cv_score)

test['flag'] = oof_test
test['flag'] = test['flag'].apply(lambda x: 0 if x < 0.5 else 1)
test[['id', 'flag']].to_csv('./result/bert_{}.csv'.format(cv_score), index=False)

# np.save("./npy/bert_test.npy", oof_test)
