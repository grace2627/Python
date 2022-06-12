import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('psm.csv')
#print(df.shape)
#print(df.columns)
#print(df.head())
#print(df.describe())

# 整理数据
# step 1: unstack row变为column， column变为row
df1 = (df
      .unstack()
      .reset_index()
      .rename(columns={'level_0':'label', 0 :'price'})
      .groupby(['label','price'])
      .size()
      .reset_index()
      .rename(columns={0:'freq'}))

df1['sum'] = df1.groupby(['label'])['freq'].transform('sum')
df1['cumsum'] = df1.groupby(['label'])['freq'].cumsum()
df1['percentage'] = df1['cumsum']/df1['sum']*100
print(df1)


df2 = df1.pivot_table('percentage', 'price', 'label')
print(df2)

# 当我们想画图，发现出现了missing value， setdefault as 0
#df3 = df2.ffill().fillna(0)
# 或者做interpolate(missing value 不多的话不需要做）
df3 = df2.interpolate().fillna(0)

# 再 inverse回来
df3['Too Cheap'] = 100 -df3['Too Cheap']
df3['Cheap'] = 100 -df3['Cheap']

df3.plot()
plt.show()