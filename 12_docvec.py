#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/6 14:55
@Author:  yanqiang
@File: 12_docvec.py
"""
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


def get_d2v_fea():
    # 构建doc2vc的训练语料
    corpus = []
    for index, text in tqdm(enumerate(texts_list)):
        label = str(index)
        corpus.append(doc2vec.TaggedDocument(text, [label]))

    # 训练doc2vec
    print("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=2, window=2, vector_size=50,
                          workers=4, alpha=0.1, max_vocab_size=400000,
                          min_alpha=0.01, dm=1)

    d2v.build_vocab(corpus)
    print("Training Doc2Vec model")
    d2v.train(corpus, total_examples=d2v.corpus_count, epochs=1)
    print("Saving trained Doc2Vec model")
    d2v.save("tmp/d2v.model")

    train_vecs = []
    test_vecs = []
    for c in train.token_text:
        data = d2v.infer_vector(c.split())
        train_vecs.append(data.tolist())
    X = np.array(train_vecs)
    for c in test.token_text:
        data = d2v.infer_vector(c.split())
        test_vecs.append(data.tolist())
    test_data = np.array(test_vecs)

    return X, test_data


X, test_data = get_d2v_fea()
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

    train_pred[val_ind, :] = sgd.predict_proba(X_val)
    test_pred += sgd.predict_proba(test_data)

    y_pred = sgd_huber.predict(X_val)
    cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')

labels = np.argmax(test_pred, axis=1)
sub['label'] = labels
sub.to_csv('result/d2v.csv', index=None)

# 训练数据预测结果
# 概率
oof_df = pd.DataFrame(train_pred)
train = pd.concat([train, oof_df], axis=1)
# 标签
labels = np.argmax(train_pred, axis=1)
train['pred'] = labels
# 分类报告
train.to_excel('result/d2v_train.xlsx', index=None)
print(classification_report(train['label'].values, train['pred'].values))
print(f1_score(train['label'].values, train['pred'].values))
