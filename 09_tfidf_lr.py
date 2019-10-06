#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/1 22:59
@Author:  yanqiang
@File: 09_tfidf_lr.py
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba


def token(text):
    """
    实现分词
    :param text:文本
    :return:
    """
    return " ".join(jieba.cut(text))


t1 = time.time()
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

column = "token_text"
n = train.shape[0]
# ngram_range：词组切分的长度范围
# max_df：可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。
# 这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。
# 如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效
# min_df：类似于max_df，不同之处在于如果某个词的document frequence小于min_df，则这个词不会被当作关键词
# use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了
# smooth_idf：idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
# sublinear_tf：默认为False，如果设为True，则替换tf为1 + log(tf)。
vec = TfidfVectorizer(ngram_range=(2, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
vec.fit(df[column])
print(vec.get_feature_names())
trn_term_doc = vec.transform(train[column])
print(type(trn_term_doc),np.array(trn_term_doc.todense()))
test_term_doc = vec.transform(test[column])
# matrix=trn_term_doc.todense().tolist()
# print(type(matrix))
# for i in matrix[0]:
#     print(i)
print(trn_term_doc.shape)
print(test_term_doc.shape)  # 提取特征的维度

y = (train["label"] - 1).astype(int)
clf = LogisticRegression(C=4, dual=True)
clf.fit(trn_term_doc, y)
test_pred = clf.predict_proba(test_term_doc)
print(test_pred)
# 生成提交结果
labels = np.argmax(test_pred, axis=1)
sub['label'] = labels
sub.to_csv('result/lr.csv', index=None)

t2 = time.time()
print("time use:", t2 - t1)
