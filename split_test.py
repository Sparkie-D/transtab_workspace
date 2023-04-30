import random

import pandas as pd

df = pd.read_csv('data\\dataset2\\data_processed.csv', index_col=0)
rows = df.shape[0]
indexs = list(range(rows))
train_len = int(len(indexs)/2) # 可重新设置训练集大小
random.shuffle(indexs)

train_idx = indexs[:train_len]
test_idx = indexs[train_len:]

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


train_df.to_csv('data\\dataset2\\train.csv')
test_df.to_csv('data\\dataset2\\test.csv')