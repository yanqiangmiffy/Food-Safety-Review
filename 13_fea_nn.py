#!/usr/bin/env python
# _*_coding:utf-8_*_
"""
@Time :    2019/10/6 17:49
@Author:  yanqiang
@File: 13_fea_nn.py
"""
from keras.layers import *
import sys
import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from gensim.models import LdaModel, LdaMulticore
from gensim import corpora
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, classification_report
from sklearn import linear_model
from gensim.models.word2vec import Word2Vec
from gensim.models import LdaModel, LdaMulticore
from gensim import corpora
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, classification_report
from sklearn import linear_model
import logging
from gensim.models import doc2vec
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba
import os
from keras.layers import *
import jieba
import multiprocessing
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
import ipykernel
import datetime


def token(text):
    """
    实现分词
    :param text:文本
    :return:
    """
    return " ".join(jieba.cut(text))


def train_w2v(text_list=None, output_vector='data/w2v.txt'):
    """
    训练word2vec
    :param text_list:文本列表
    :param output_vector:词向量输出路径
    :return:
    """
    print("正在训练词向量。。。")
    corpus = [text.split() for text in text_list]
    model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
    # 保存词向量
    model.wv.save_word2vec_format(output_vector, binary=False)


# sample.csv
# test_new.csv
# train.csv
train = pd.read_csv('data/train.csv', sep='\t')
test = pd.read_csv('data/test_new.csv')
sub = pd.read_csv('data/sample.csv')

# 数据处理 复制label为1文本
# index = train.label == 1
# print(index)
# train[index]
# train.['comment'] = train[index]['comment'].apply(lambda x: x +'。'+ x)
# 全量数据
train['id'] = [i for i in range(len(train))]
test['label'] = [-1 for i in range(len(test))]
df = pd.concat([train, test], sort=False)
df['token_text'] = df['comment'].apply(lambda x: token(x))
texts = df['token_text'].values.tolist()
# train_w2v(texts)

# 构建词汇表
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print("词语数量个数：{}".format(len(word_index)))

# 数据
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100

sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# 类别编码
X = data[:len(train)]
x_test = data[len(train):]
y = to_categorical(train['label'].values)
y = y.astype(np.int32)

bert_x_train = np.load(open('tmp/feas/bert_x_train.npy', 'rb'))
bert_x_test = np.load(open('tmp/feas/bert_x_test.npy', 'rb'))
print("bert_x_train.shape:", bert_x_train.shape)
print("bert_x_test.shape:", bert_x_test.shape)

d2v_50_x_train = np.load(open('tmp/feas/d2v_50_x_train.npy', 'rb'))
d2v_50_x_test = np.load(open('tmp/feas/d2v_50_x_test.npy', 'rb'))
print("d2v_50_x_train.shape:", d2v_50_x_train.shape)
print("d2v_50_x_test.shape:", d2v_50_x_test.shape)

w2v_50_x_train = np.load(open('tmp/feas/w2v_50_x_train.npy', 'rb'))
w2v_50_x_test = np.load(open('tmp/feas/w2v_50_x_test.npy', 'rb'))
print("w2v_50_x_train.shape:", w2v_50_x_train.shape)
print("w2v_50_x_test.shape:", w2v_50_x_test.shape)

lda_x_train = np.load(open('tmp/feas/lda_x_train.npy', 'rb'))
lda_x_test = np.load(open('tmp/feas/lda_x_test.npy', 'rb'))
print("lda_x_train.shape:", lda_x_train.shape)
print("lda_x_test.shape:", lda_x_test.shape)

features_x_train = np.hstack((d2v_50_x_train, w2v_50_x_train, lda_x_train))
features_x_test = np.hstack((d2v_50_x_test, w2v_50_x_test, lda_x_test))
print("features_x_train.shape:", features_x_train.shape)
print("features_x_test.shape:", features_x_test.shape)

af_x_train = np.load(open('tmp/feas/af_x_train.npy', 'rb'))
af_x_test = np.load(open('tmp/feas/af_x_test.npy', 'rb'))
print("af_x_train.shape:", af_x_train.shape)
print("af_x_test.shape:", af_x_test.shape)

# 创建embedding_layer
def create_embedding(word_index, w2v_file):
    """

    :param word_index: 词语索引字典
    :param w2v_file: 词向量文件
    :return:
    """
    embedding_index = {}
    f = open(w2v_file, 'r', encoding='utf-8')
    next(f)  # 下一行
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print("Total %d word vectors in w2v_file" % len(embedding_index))

    embedding_matrix = np.random.random(size=(len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH, trainable=True)
    return embedding_layer


def create_text_cnn():
    # 序列输入
    seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = create_embedding(word_index, 'data/w2v.txt')
    embedding_input = embedding_layer(seq_input)
    l_lstm = Bidirectional(CuDNNLSTM(128))(embedding_input)

    # bert输入
    bert_input = Input(shape=(768,))
    bert_dense = BatchNormalization()(bert_input)
    bert_dense = Dense(64, activation='relu')(bert_dense)

    # tfidf lda w2v doc2vec
    features_input = Input(shape=(108,))
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(128, activation='relu')(features_dense)

    af_input = Input(shape=(62,))
    af_dense = BatchNormalization()(af_input)
    af_dense = Dense(32, activation='relu')(af_dense)

    merged = concatenate([l_lstm, bert_dense, features_dense,af_dense])
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)

    output = Dense(2, activation='softmax')(merged)
    merge_model = Model(inputs=[seq_input, bert_input, features_input,af_input], outputs=output)
    return merge_model


train_pred = np.zeros((len(train), 2))
test_pred = np.zeros((len(test), 2))
#
skf = StratifiedKFold(n_splits=5, random_state=52, shuffle=True)
for i, (train_index, valid_index) in enumerate(skf.split(X, train['label'])):
    print("n@:{}fold".format(i + 1))
    X_train = X[train_index]
    X_valid = X[valid_index]

    X_bert_train = bert_x_train[train_index]
    X_bert_valid = bert_x_train[valid_index]

    X_fea_train = features_x_train[train_index]
    X_fea_valid = features_x_train[valid_index]

    X_af_train = af_x_train[train_index]
    X_af_valid = af_x_train[valid_index]

    y_tr = y[train_index]
    y_val = y[valid_index]

    #
    model = create_text_cnn()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    checkpoint = ModelCheckpoint(filepath='models/feann_text_{}.h5'.format(i + 1),
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True)
    tr_data=[X_train, X_bert_train, X_fea_train,X_af_train]
    va_data=[X_valid, X_bert_valid, X_fea_valid,X_af_valid]
    history = model.fit(tr_data, y_tr,
                        validation_data=(va_data, y_val),
                        epochs=10, batch_size=32,
                        callbacks=[checkpoint])

    # model.load_weights('models/cnn_text.h5')
    train_pred[valid_index, :] = model.predict([X_valid, X_bert_valid, X_fea_valid,X_af_valid])
    test_pred += model.predict([x_test, bert_x_test, features_x_test,af_x_test])

labels = np.argmax(test_pred, axis=1)
sub['label'] = labels
sub.to_csv('result/fea_nn.csv', index=None)

# 训练数据预测结果
# 概率
oof_df = pd.DataFrame(train_pred)
train = pd.concat([train, oof_df], axis=1)
# 标签
labels = np.argmax(train_pred, axis=1)
train['pred'] = labels
# 分类报告
train.to_excel('result/fea_nn.xlsx', index=None)
print(classification_report(train['label'].values, train['pred'].values))
print(f1_score(train['label'].values, train['pred'].values))
