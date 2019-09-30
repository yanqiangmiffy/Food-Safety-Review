import pandas as pd
import numpy as np
df = pd.read_csv('xx_all_data.csv')
print(df.columns)
df['id']=[i for i in range(len(df))]
df['pred']=np.argmax(df[['0','1']].values,axis=1)
df.to_excel('result/bert_train.xlsx',index=None)
