#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/6 14:21
@Author:  yanqiang
@File: 11_w2v.py
"""
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


def token(text):
    """
    实现分词
    :param text:文本
    :return:
    """
    return " ".join(jieba.cut(text))


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
train['token_text'] = train['comment'].apply(lambda x: token(x))
test['token_text'] = test['comment'].apply(lambda x: token(x))
df['token_text'] = df['comment'].apply(lambda x: token(x))
texts = df['token_text'].values.tolist()
texts_list = [[word for word in doc.split()] for doc in texts]


def get_w2v_feas():
    ## 训练1h 13min 32s
    ## 数据集4kw train  2kw test
    print("正在准备训练数据集..")
    w2v_model = Word2Vec(min_count=1, size=300, window=10, workers=8, iter=10)
    w2v_model.build_vocab(texts_list)
    print("正在训练word2vec....")
    w2v_model.train(texts_list, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
    w2v_model.save('tmp/w2v_300d.model')


    train_vecs = []
    test_vecs = []
    for c in train.token_text:
        data = np.mean([w2v_model[word] for word in c.split()], axis=0).reshape(1, -1)
        train_vecs.append(list(data[0]))
    X = np.array(train_vecs)
    for c in test.token_text:
        data = np.mean([w2v_model[word] for word in c.split()], axis=0).reshape(1, -1)
        test_vecs.append(list(data[0]))
    test_data = np.array(test_vecs)

    return X, test_data


X, test_data = get_w2v_feas()
print(X)
print(test_data)
y = np.array((train['label']).values)

skf = StratifiedKFold(n_splits=5, random_state=52, shuffle=True)
cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1, = [], [], []

train_pred = np.zeros((len(train), 2))
test_pred = np.zeros((len(test), 2))
print(train_pred)
for train_ind, val_ind in skf.split(X, y):
    # Assign CV IDX
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind]

    # Logisitic Regression
    lr = LogisticRegression(
        class_weight='balanced',
        solver='newton-cg',
        fit_intercept=True
    ).fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    print(y_pred)
    cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))

    # Logistic Regression Mini-Batch SGD
    sgd = linear_model.SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        loss='log',
        class_weight='balanced'
    ).fit(X_train, y_train)

    y_pred = sgd.predict(X_val)
    cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

    # SGD Modified Huber
    sgd_huber = linear_model.SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        alpha=20,
        loss='modified_huber',
        class_weight='balanced'
    ).fit(X_train, y_train)

    train_pred[val_ind, :] = sgd_huber.predict_proba(X_val)
    test_pred += sgd_huber.predict_proba(test_data)

    y_pred = sgd_huber.predict(X_val)
    cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')

labels = np.argmax(test_pred, axis=1)
sub['label'] = labels
sub.to_csv('result/w2v.csv', index=None)

# 训练数据预测结果
# 概率
oof_df = pd.DataFrame(train_pred)
train = pd.concat([train, oof_df], axis=1)
# 标签
labels = np.argmax(train_pred, axis=1)
train['pred'] = labels
# 分类报告
train.to_excel('result/w2v_train.xlsx', index=None)
print(classification_report(train['label'].values, train['pred'].values))
print(f1_score(train['label'].values, train['pred'].values))
