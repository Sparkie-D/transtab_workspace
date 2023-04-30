import pandas as pd

path = 'E:\\NJU\\JuniorII\\lamda5\\TransTab\\testProject\\data\\dataset2'

# 去除target中的nan
df = pd.read_csv(path+"\\data_processed.csv")
# df = df.dropna(subset=['target_label'])
# df.to_csv(path+"\\data_processed.csv", index=True)

# 类型转换
bin_file = path + '\\binary_feature.txt'
bin_names = []
with open(bin_file) as f:
    bin_names = f.readlines()
bin_names = [item.replace('\n', '') for item in bin_names]

df[bin_names] = df[bin_names].astype('int64', errors='ignore')

print(df)