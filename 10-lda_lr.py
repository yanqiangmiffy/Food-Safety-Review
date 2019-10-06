from gensim.models import LdaModel, LdaMulticore
from gensim import corpora
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba
from sklearn.model_selection import KFold,StratifiedKFold
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
data = [[word for word in doc.split()] for doc in texts]

dct = corpora.Dictionary(data)
corpus = [dct.doc2bow(line) for line in data]

if __name__ == '__main__':
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

    # See the topics
    lda_model.print_topics(-1)

    for c in lda_model[corpus[5:8]]:
        print("Document Topics      : ", c[0])  # [(Topics, Perc Contrib)]
        print("Word id, Topics      : ", c[1][:3])  # [(Word id, [Topics])]
        print("Phi Values (word id) : ", c[2][:2])  # [(Word id, [(Topic, Phi Value)])]
        print("Word, Topics         : ", [(dct[wd], topic) for wd, topic in c[1][:2]])  # [(Word, [Topics])]
        print("Phi Values (word)    : ", [(dct[wd], topic) for wd, topic in c[2][:2]])  # [(Word, [(Topic, Phi Value)])]
        print("------------------------------------------------------\n")

    train_vecs = []
    for i in range(len(train)):
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        # topic_vec.extend([train.iloc[i].real_counts]) # counts of reviews for restaurant
        topic_vec.extend([len(train.iloc[i].comment)])  # length review
        train_vecs.append(topic_vec)
    print(train_vecs)
    X = np.array(train_vecs)
    y = np.array((train['label']).values)
    print(y)
    print(len(X), len(y))
    test_vecs = []
    for i in range(len(test)):
        i = i + len(train)
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        # topic_vec.extend([train.iloc[i].real_counts]) # counts of reviews for restaurant
        topic_vec.extend([len(df.iloc[i].comment)])  # length review
        test_vecs.append(topic_vec)
    test_data = np.array(test_vecs)
    print(test.ndim)

    train_pred = np.zeros((len(train), 2))
    test_pred = np.zeros((len(test), 2))

    kf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1, = [], [], []

    for train_ind, val_ind in kf.split(X, y):
        # Assign CV IDX
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        # Scale Data
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_val_scale = scaler.transform(X_val)

        # Logisitic Regression
        lr = LogisticRegression(
            class_weight='balanced',
            solver='newton-cg',
            fit_intercept=True
        ).fit(X_train_scale, y_train)

        y_pred = lr.predict(X_val_scale)
        print(y_pred)
        cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))

        # Logistic Regression Mini-Batch SGD
        sgd = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            loss='log',
            class_weight='balanced'
        ).fit(X_train_scale, y_train)

        y_pred = sgd.predict(X_val_scale)
        cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

        # SGD Modified Huber
        sgd_huber = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            alpha=20,
            loss='modified_huber',
            class_weight='balanced'
        ).fit(X_train_scale, y_train)

        train_pred[val_ind, :] = lr.predict_proba(X_val)
        test_pred += lr.predict_proba(test_data)
        print(lr.predict_proba(test_data))
        y_pred = sgd_huber.predict(X_val_scale)
        cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

    print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
    print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
    print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')

    labels = np.argmax(test_pred, axis=1)
    sub['label'] = labels
    sub.to_csv('result/lda.csv', index=None)

    # 训练数据预测结果
    # 概率
    oof_df = pd.DataFrame(train_pred)
    train = pd.concat([train, oof_df], axis=1)
    # 标签
    labels = np.argmax(train_pred, axis=1)
    train['pred'] = labels
    # 分类报告
    train.to_excel('result/lda_train.xlsx', index=None)
    print(classification_report(train['label'].values, train['pred'].values))
    print(f1_score(train['label'].values, train['pred'].values))
