#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/6 16:57
@Author:  yanqiang
@File: multi_input.py
"""
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
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba
import os


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

train_size = len(train)
test_size = len(test)
all_size = len(df)

# for gensim
data = [[word for word in doc.split()] for doc in texts]
dct = corpora.Dictionary(data)
corpus = [dct.doc2bow(line) for line in data]


def create_bert_input():
    print('加载bert预训练模型并提取cls embedding')

    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    else:
        from keras_bert.datasets import get_pretrained, PretrainedList
        model_path = get_pretrained(PretrainedList.chinese_base)
    paths = get_checkpoint_paths(model_path)
    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=None)
    model.summary(line_length=120)
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(token_dict)

    # base = 'D:/data/bert/chinese_wwm_ext_L-12_H-768_A-12/'
    # config_path = base + 'bert_config.json'
    # checkpoint_path = base + 'bert_model.ckpt'
    # dict_path = base + 'vocab.txt'
    # model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # model.summary(line_length=120)
    # token_dict = load_vocabulary(dict_path)
    # tokenizer = Tokenizer(token_dict)

    all_vecs = []
    for text in tqdm(df.comment):
        indices, segments = tokenizer.encode(first=text, max_len=10)
        predicts = model.predict([np.array([indices]), np.array([segments])])[0]
        all_vecs.append(predicts[0].tolist())
    bert_x_train = np.array(all_vecs[:train_size])
    bert_x_test = np.array(all_vecs[train_size:])
    print('Shape of train data tensor:', bert_x_train.shape)
    print('Shape of test data tensor:', bert_x_test.shape)
    np.save("tmp/feas/bert_x_train.npy", bert_x_train)
    np.save("tmp/feas/bert_x_test.npy", bert_x_test)


def create_tfidf_input():
    column = 'token_text'
    ngram_ranges = [(1, 1), (2, 2), (3, 3)]
    for nr in ngram_ranges:
        vec = TfidfVectorizer(ngram_range=nr, min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
        vec.fit(df[column])
        print(vec.get_feature_names())
        trn_term_doc = vec.transform(train[column])
        test_term_doc = vec.transform(test[column])

        tfidf_x_train = np.array(trn_term_doc.todense())
        tfidf_x_test = np.array(test_term_doc.todense())
        print('Shape of train data tensor:', tfidf_x_train.shape)
        print('Shape of test data tensor:', tfidf_x_test.shape)
        np.save("tmp/feas/tfidf_{}_x_train.npy".format(str(nr)), tfidf_x_train)
        np.save("tmp/feas/tfidf_{}_x_test.npy".format(str(nr)), tfidf_x_test)


def create_lda_input():
    # 训练lda
    num_topics = 7
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dct,
                             random_state=100,
                             num_topics=num_topics,
                             passes=10,
                             chunksize=1000,
                             batch=False,
                             alpha='asymmetric',
                             decay=0.5,
                             offset=64,
                             eta=None,
                             eval_every=0,
                             iterations=100,
                             gamma_threshold=0.001,
                             per_word_topics=True)

    # save the model
    lda_model.save('tmp/lda_model.model')
    all_vecs = []
    for i in range(len(df)):
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        topic_vec.extend([len(df.iloc[i].comment)])  # length review
        all_vecs.append(topic_vec)

    lda_x_train = np.array(all_vecs[:train_size])
    lda_x_test = np.array(all_vecs[train_size:])
    print('Shape of train data tensor:', lda_x_train.shape)
    print('Shape of test data tensor:', lda_x_test.shape)
    np.save("tmp/feas/lda_x_train.npy", lda_x_train)
    np.save("tmp/feas/lda_x_test.npy", lda_x_test)


def create_w2v_input():
    sizes = [50, 100, 300]
    for size in sizes:
        w2v_model = Word2Vec(min_count=1, size=size, window=10, workers=8, iter=10)
        w2v_model.build_vocab(texts_list)
        print("正在训练word2vec....")
        w2v_model.train(texts_list, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
        w2v_model.save('tmp/w2v_300d.model')

        all_vecs = []
        for c in df.token_text:
            tmp = np.mean([w2v_model[word] for word in c.split()], axis=0).reshape(1, -1)
            all_vecs.append(list(tmp[0]))

        w2v_x_train = np.array(all_vecs[:train_size])
        w2v_x_test = np.array(all_vecs[train_size:])
        print('Shape of train data tensor:', w2v_x_train.shape)
        print('Shape of test data tensor:', w2v_x_test.shape)
        np.save("tmp/feas/w2v_{}_x_train.npy".format(str(size)), w2v_x_train)
        np.save("tmp/feas/w2v_{}_x_test.npy".format(str(size)), w2v_x_test)


def create_d2v_input():
    d_corpus = []
    for index, text in tqdm(enumerate(texts_list)):
        label = str(index)
        d_corpus.append(doc2vec.TaggedDocument(text, [label]))

    # 训练doc2vec
    sizes = [50, 100, 200]
    print("Building Doc2Vec vocabulary")
    for size in sizes:
        d2v = doc2vec.Doc2Vec(min_count=1, window=2, vector_size=size,
                              workers=os.cpu_count() - 1, alpha=0.1, max_vocab_size=400000,
                              min_alpha=0.01, dm=1)

        d2v.build_vocab(d_corpus)
        print("Training Doc2Vec model")
        d2v.train(d_corpus, total_examples=d2v.corpus_count, epochs=1)
        # print("Saving trained Doc2Vec model")
        # d2v.save("tmp/d2v.model")

        all_vecs = []
        for c in df.token_text:
            data = d2v.infer_vector(c.split())
            all_vecs.append(data.tolist())
        d2v_x_train = np.array(all_vecs[:train_size])
        d2v_x_test = np.array(all_vecs[train_size:])
        print('Shape of train data tensor:', d2v_x_train.shape)
        print('Shape of test data tensor:', d2v_x_test.shape)
        np.save("tmp/feas/d2v_{}_x_train.npy".format(str(size)), d2v_x_train)
        np.save("tmp/feas/d2v_{}_x_test.npy".format(str(size)), d2v_x_test)


def create_artificial_feature():
    def load_neg_words():
        pass

    def load_pos_words():
        pass


if __name__ == '__main__':
    create_bert_input()
    create_tfidf_input()
    create_w2v_input()
    create_d2v_input()
    # create_lda_input()
