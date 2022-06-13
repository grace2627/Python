import pandas as pd
import numpy as np

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
# print(res.summary())

print(res.params.items)
print(res.pvalues)

# create a new dataframe with name, coefficient, and pvalues
df_res = pd.DataFrame(
    {
        'name': res.params.keys(),
        'coeff': res.params.values,
        'pvalue':res.pvalues
    }
)
# print(df_res)

# plot
pd.options.display.float_format = '{:.2f}'.format
# print(df_res)

df_res['abs_coeff'] = np.abs(df_res['coeff'])
df_res['sig_95'] = df_res['pvalue']< 0.05
df_res['color'] = [ 'b' if x else 'r' for x in df_res['sig_95']]
df_res =df_res.sort_values(by='abs_coeff', ascending=True)
f, ax = plt.subplots(figsize = (14, 8))
plt.title('part worth')
pwu = df_res['pvalue']
xbar = np.arange(len(pwu))

plt.barh(xbar, df_res['coeff'], color=df_res['color'])
plt.yticks(xbar, labels=df_res['name'])

# print(df_res)
# plt.show()

# %%%


feature_range = dict()
for key, coeff in res.params.items():
    feature = key.split('_')[0]
    # 把每个attribute里面的小项给分解出来
    if feature not in feature_range:
        feature_range[feature] = list()

    feature_range[feature].append(coeff)
print(feature_range)

feature_importance =\
    { key: max(value)-min(value) for key, value in feature_range.items()}
print(feature_importance)

total_importance = sum(feature_importance.values())
feature_relative_importance = {
key: round(value/total_importance, 3) * 100 for key, value in feature_importance.items()
}
print(feature_relative_importance)

df_feature_imp = pd.DataFrame(list(feature_importance.items()),
                              columns=['feature', 'importance'])\
    .sort_values(by='importance', ascending=False)

df_feature_relativeimp = pd.DataFrame(list(feature_relative_importance.items()),
                              columns=['feature', 'importance'])\
    .sort_values(by='importance', ascending=False)

f, (ax1, ax2) = plt.subplots(ncols=2, nrows = 1, figsize = (14, 8))

sns.barplot(data = df_feature_imp, x='feature', y='importance', ax = ax1)
sns.barplot(data = df_feature_relativeimp, x='feature', y='importance', ax = ax2)
plt.show()