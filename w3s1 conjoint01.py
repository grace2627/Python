import pandas as pd

df = pd.read_csv('candidate_1.tab.txt', delimiter= '\t')
'''print(df.columns)
print(df.shape)
print(df.describe())
print(df['religion'].unique())
print(df.dtypes)

# 看看有没有missing value
print(df.isnull().sum())
# rating 有十个missing value
'''
# 先做preprocessing， 把不相关的去掉
df_input = df[['education', 'religion', 'research_area', 'professional',
       'pricing_group', 'race', 'age_group', 'gender']]
# print(df_input)

import seaborn as sns
import matplotlib.pyplot as plt

'''
# 看看男女，宗教人数
sns.countplot(data=df_input, x = 'gender', hue = 'religion')
plt.show()
'''

print(pd.get_dummies(data= df_input, columns = df_input.columns))
# 从8个col 变成40个col， 在降维

# label encoding
# onehot encoding

import statsmodels.api as sm

olsmodel = sm.OLS(df['selected'], pd.get_dummies(data= df_input, columns = df_input.columns))
res=olsmodel.fit()
print(res.summary())