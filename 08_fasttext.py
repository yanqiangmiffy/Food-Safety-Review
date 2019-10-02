#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/1 14:29
@Author:  yanqiang
@File: 08_fasttext.py
"""
import os
import jieba
import fasttext
from fasttext import train_supervised
import pandas as pd
import numpy as np


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
df['token_text'] = df['comment'].apply(lambda x: token(x))

train_size = len(train)
with open('tmp/train.txt', 'w', encoding='utf-8') as f:
    for text, label in zip(df[:train_size].token_text, df[:train_size].label):
        f.write('__label__' + str(int(label)) + ' ' + text + '\n')
with open('tmp/test.txt', 'w', encoding='utf-8') as f:
    for text in df[train_size:].token_text:
        f.write(text + '\n')


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


train_data = os.path.join(os.getenv('', 'tmp'), 'train.txt')
test_data = os.path.join(os.getenv('', 'tmp'), 'test.txt')

print(train_data)
model = train_supervised(input=train_data, epoch=50, lr=1.0, wordNgrams=5, verbose=2, minCount=1)
print(*model.test(train_data))

with open('tmp/test.txt', 'r', encoding='utf-8') as f:
    test_txt = f.read().split('\n')
test_txt = test_txt[:len(test_txt) - 1]
res, probs = (model.predict(test_txt, k=1))
labels = []
for label in res:
    if label[0].endswith('0'):
        labels.append(0)
    if label[0].endswith('1'):
        labels.append(1)

sub['label'] = labels
sub.to_csv('result/fasttext.csv', index=None)
