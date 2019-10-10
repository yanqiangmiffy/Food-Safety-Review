import pandas as pd
import numpy as np

bert = pd.read_csv('./result/bert.csv')
fea_nn = pd.read_csv('./result/fea_nn.csv')
bi_rnn = pd.read_csv('./result/bi_rnn.csv')
capsule = pd.read_csv('./result/capsule.csv')
cnn = pd.read_csv('./result/cnn.csv')
cnn1 = pd.read_csv('./result/cnn1.csv')
fasttext = pd.read_csv('./result/fasttext.csv')
han = pd.read_csv('./result/han.csv')
lr = pd.read_csv('./result/lr.csv')
rcnnv = pd.read_csv('./result/rcnnv.csv')
rnn = pd.read_csv('./result/rnn.csv')
bi_gru = pd.read_csv('./result/bi_gru.csv')

concat = [
    bert,
    bert[['label']],bert[['label']],bert[['label']],bert[['label']],
    bi_rnn[['label']],
    capsule[['label']],
    cnn[['label']],
    # cnn1[['label']],
    # fasttext[['label']],
    han[['label']],
    # lr[['label']],
    rcnnv[['label']],
    rnn[['label']],
    bi_gru[['label']],
    fea_nn[['label']]
]
bert = pd.concat(concat, axis=1)

b_value = bert.values
b_value = np.delete(b_value, [0], axis=1)
b_value = np.sum(b_value, axis=1)
b_value = np.array([0 if i <= int(len(concat) / 2) else 1 for i in b_value])

id = bert.values[:, 0]

stack = pd.DataFrame({'id': id, 'label': b_value})

stack.to_csv('result/stack.csv', index=None)
