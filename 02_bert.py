# coding:utf-8
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
import codecs
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras.callbacks import *
import warnings
import numpy as np
import sys
import re
import time

np.random.seed(42)
warnings.filterwarnings('ignore')


class data_generator:
    def __init__(self, data, batch_size=128, shuffle=True, textmodel=1):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.textmodel = textmodel
        print('textmodel', self.textmodel)
        print('shuffle', self.shuffle)
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, X3, X4, X5, X6, Y = [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                base_len = len(d[0])
                if self.textmodel == 3:
                    if base_len <= 510:
                        text = d[0][:510]
                    else:
                        text1 = d[0][:128]
                        base_len_128 = base_len - 128
                        text2 = d[0][base_len_128 - 382 + 128:]
                        text = text1 + text2
                    assert len(text) <= 510
                elif self.textmodel == 2:
                    if base_len <= 510:
                        text = d[0][:510]
                    else:
                        offset = base_len - 510
                        text = d[0][offset:]
                elif self.textmodel == 1:
                    text = d[0][:30]
                else:
                    text = d[0]

                if self.textmodel in [1, 2, 3]:
                    x1, x2 = tokenizer.encode(first=text)
                    y = d[1]
                    X1.append(x1)
                    X2.append(x2)
                    Y.append(to_categorical(y, 2))
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        Y = seq_padding(Y)
                        yield [X1, X2], Y
                        [X1, X2, Y] = [], [], []
                else:
                    text1 = text[:510]
                    text2 = text[510:1020]
                    # text3 = text[1020:1530]
                    x1, x2 = tokenizer.encode(first=text1)
                    x3, x4 = tokenizer.encode(first=text2)
                    # x5, x6 = tokenizer.encode(first=text3)
                    y = d[1]
                    X1.append(x1)
                    X2.append(x2)
                    X3.append(x3)
                    X4.append(x4)
                    # X5.append(x5)
                    # X6.append(x6)
                    Y.append(to_categorical(y, 2))
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        X3 = seq_padding(X3)
                        X4 = seq_padding(X4)
                        # X5 = seq_padding(X5)
                        # X6 = seq_padding(X6)
                        Y = seq_padding(Y)
                        yield [X1, X2, X3, X4], Y
                        [X1, X2, X3, X4, Y] = [], [], [], [], []


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)

    p = Dense(2, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    return model


def get_model_mut_input():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x1 = bert_model([x1_in, x2_in])
    x1 = Lambda(lambda x: x[:, 0])(x1)

    x3_in = Input(shape=(None,))
    x4_in = Input(shape=(None,))
    x2 = bert_model([x3_in, x4_in])
    x2 = Lambda(lambda x: x[:, 0])(x2)

    # x5_in = Input(shape=(None,))
    # x6_in = Input(shape=(None,))
    # x3 = bert_model([x5_in, x6_in])
    # x3 = Lambda(lambda x: x[:, 0])(x3)

    x = concatenate([x1, x2], axis=-1)

    p = Dense(2, activation='softmax')(x)

    model = Model([x1_in, x2_in, x3_in, x4_in], p)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    return model


def pre_process(text):
    text = str(text).replace('\\u3000', '')
    pattern = "[\：\、\•\“\”\【\】\!\！\?\？\,\，\。0-9\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    results = regex.findall(str(text))
    return ' '.join(results)


maxlen = 100
base = 'D:/data/bert/chinese_wwm_ext_L-12_H-768_A-12/'
config_path = base + 'bert_config.json'
checkpoint_path = base + 'bert_model.ckpt'
dict_path = base + 'vocab.txt'
# if len(sys.argv) == 2:
#     model_path = sys.argv[1]
# else:
#     from keras_bert.datasets import get_pretrained, PretrainedList
#
#     model_path = get_pretrained(PretrainedList.chinese_base)
# paths = get_checkpoint_paths(model_path)
# config_path = paths.config
# checkpoint_path = paths.checkpoint
# dict_path = paths.vocab

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = OurTokenizer(token_dict)


def run_cv():
    start_time = time.time()
    # 1 读取数据
    label_map = {1: 1, 0: 0}
    label_map_r = {1: 1, 0: 0}
    train = pd.read_csv('data/train.csv', sep='\t')
    test = pd.read_csv('data/test_new.csv')
    sub = pd.read_csv('data/sample.csv')

    # print(test[test['text'].isin(train['text'].unique())])

    # all_data['text'] = all_data['title'].astype(str) + ' ' + all_data['content'].astype(str)
    train_data = train[['comment', 'label']]
    train_data['comment'] = train_data['comment'].apply(lambda x: ''.join(str(x).split(' ')))
    train_data.columns = ['comment', 'label']

    test = test[['id', 'comment']]
    test['comment'] = test['comment'].apply(lambda x: ''.join(str(x).split(' ')))
    test.columns = ['id', 'comment']

    test_data = []
    print('make bert data')
    for d in test.values:
        # x1, x2 = tokenizer.encode(first=d[1])
        test_data.append((d[1], 0))
    print(len(test_data))
    submit_D = data_generator(test_data, batch_size=256, shuffle=False, textmodel=1)
    del test_data

    xx_all_data = train_data.copy()

    n_flod = 5
    oof = np.zeros((train_data.shape[0], 2))
    oof_sub = np.zeros((test.shape[0], 2))
    skf = StratifiedKFold(n_splits=n_flod, random_state=42, shuffle=True)
    for j, (tr_idx, te_idx) in enumerate(skf.split(train_data, train_data['label'].values)):
        print('bert for ', j)
        K.clear_session()
        X_train = train_data.iloc[tr_idx][['comment', 'label']].values
        X_valid = train_data.iloc[te_idx][['comment', 'label']].values

        bert_model = get_model()
        if j == 0:
            print(bert_model.summary())
        checkpointer = ModelCheckpoint(filepath="models/checkpoint_%d.hdf5" % (j),
                                       monitor='val_acc', verbose=True,
                                       save_best_only=True, mode='auto')
        early = EarlyStopping(monitor='val_acc', patience=1, verbose=0, mode='auto')

        print('data_generator', j)
        train_D = data_generator(X_train, batch_size=4, shuffle=True, textmodel=1)
        valid_D = data_generator(X_valid, batch_size=32, shuffle=False, textmodel=1)
        del X_train, X_valid
        # if j > 8:
        bert_model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[checkpointer, early],
            verbose=True
        )
        # bert_model.load_weights("data/checkpoint_%d.hdf5" % (j))
        # bert_model.save_weights("models/checkpoint_%d.hdf5" % (j))
        test_y = bert_model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        oof[te_idx] = test_y
        del train_D, valid_D

        oof_sub = oof_sub + bert_model.predict_generator(submit_D.__iter__(), steps=len(submit_D), verbose=1) / n_flod
        del bert_model
        print('finish skflod', j + 1)
    # del test_data
    # 原始数据
    oof_df = pd.DataFrame(oof)
    xx_all_data = pd.concat([xx_all_data, oof_df], axis=1)
    xx_all_data.to_csv('./xx_all_data.csv', index=False)

    xx_cv = accuracy_score(train_data['label'].values, np.argmax(oof, axis=1))
    print(xx_cv)

    result = pd.DataFrame()
    result['id'] = test['id']
    xx_result = result.copy()

    oof_sub_df = pd.DataFrame(oof_sub)
    xx_result = pd.concat([xx_result, oof_sub_df], axis=1)
    xx_result.to_csv('./xx_result.csv', index=False)
    del test
    result['labels'] = np.argmax(oof_sub, axis=1)
    result['label'] = result['labels'].map(label_map_r)
    print(result)

    result[['id', 'label']].to_csv('./result/baseline_bert_{}_{}.csv'.format(n_flod, str(np.mean(xx_cv)).split('.')[1]),
                                   index=False)
    end_time = time.time()
    print("共耗时：", end_time - start_time)


run_cv()
