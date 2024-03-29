#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(suppress = True)


# # 第九章
# 
# ## 零、练一练
# 
# 
# ```{admonition} 练一练
# 请构造一个无序类别元素构成的序列，通过cat对象完成增删改查的操作。
# ```

# In[2]:


s = pd.Series(list("ABCD")).astype("category")
s


# In[3]:


s.cat.categories


# In[4]:


s = s.cat.add_categories('E')
s


# In[5]:


s = s.cat.remove_categories('E')
s


# In[6]:


s.cat.rename_categories({'D':'E'})


# ```{admonition} 练一练
# 请构造两组因不同原因（索引不一致和Series类别不一致）而导致无法比较的有序类别Series。
# ```

# In[7]:


s1 = pd.Series(list("ABCD")).astype("category")
s2 = pd.Series(list("ABCD"), index=[1,2,3,5]).astype("category")
s1 == s2


# In[8]:



s1 = pd.Series(list("ABCD")).astype("category")
s2 = pd.Series(list("ABCD")).astype("category")
s2 = s2.cat.add_categories('E')
s1 == s2


# ## 一、统计未出现的类别
# 
# crosstab()函数是一种特殊的变形函数。在默认参数下，它能够对两个列中元素组合出现的频数进行统计汇总：

# In[9]:


df = pd.DataFrame({'A':['a','b','c','a'],
                   'B':['cat','cat','dog','cat']})
pd.crosstab(df.A, df.B)


# 但事实上，有些列存储的是分类变量，列中并不一定包含所有的类别，此时如果想要对这些未出现的类别在crosstab()结果中也进行汇总，则可以指定dropna参数为False：

# In[10]:


df.B = df.B.astype('category').cat.add_categories('sheep')
pd.crosstab(df.A, df.B, dropna=False)


# 请实现1个带有dropna参数的my_crosstab()函数来完成上面的功能。
# 
# ```text
# 【解答】
# ```

# In[11]:


def my_crosstab(s1, s2, dropna=True):
    if s1.dtype.name == 'category' and not dropna:
        idx1 = s1.cat.categories
    else:
        idx1 = s1.unique()
    if s2.dtype.name == 'category' and not dropna:
        idx2 = s2.cat.categories
    else:
        idx2 = s2.unique()
    res = pd.DataFrame(
        np.zeros((idx1.shape[0], idx2.shape[0])),
        index=idx1,
        columns=idx2)
    for i, j in zip(s1, s2):
        res.at[i, j] += 1
    res = res.rename_axis(index=s1.name, columns=s2.name).astype('int')
    return res


# In[12]:


df = pd.DataFrame({'A':['a','b','c','a'],
                   'B':['cat','cat','dog','cat']})
df.B = df.B.astype('category').cat.add_categories('sheep')
my_crosstab(df.A, df.B)
my_crosstab(df.A, df.B, dropna=False)


# ## 二、钻石数据的类别构造
# 
# 现有一份关于钻石的数据集，其中carat、cut、clarity和price分别表示克拉重量、切割质量、纯净度和价格：

# In[13]:


df = pd.read_csv('data/ch9/diamonds.csv')
df.head(3)


# - 分别对df.cut在object类型和category类型下使用nunique()函数，并比较它们的性能。
# - 钻石的切割质量可以分为五个等级，由次到好分别是Fair、Good、Very Good、Premium、Ideal，纯净度有八个等级，由次到好分别是I1、SI2、SI1、VS2、VS1、VVS2、VVS1、IF，请对切割质量按照由好到次的顺序排序，相同切割质量的钻石，按照纯净度进行由次到好的排序。
# - 分别采用两种不同的方法，把cut和clarity这两列按照由好到次的顺序，映射到从0到n-1的整数，其中n表示类别的个数。
# - 对每克拉的价格分别按照分位数（q=[0.2, 0.4, 0.6, 0.8]）与[1000, 3500, 5500, 18000]割点进行分箱得到五个类别Very Low、Low、Mid、High、Very High，并把按这两种分箱方法得到的category序列依次添加到原表中。
# - 在第4问中按整数分箱得到的序列中，是否出现了所有的类别？如果存在没有出现的类别请把该类别删除。
# - 对第4问中按分位数分箱得到的序列，求序列元素所在区间的左端点值和长度。
# 
# ```text
# 【解答】
# ```
# 
# - 1

# In[14]:


df = pd.read_csv('data/ch9/diamonds.csv')
s_obj, s_cat = df.cut, df.cut.astype('category')


# ```python
# %timeit -n 30 s_obj.nunique()
# ```

# In[15]:


print("2.57 ms ± 161 µs per loop (mean ± std. dev. of 7 runs, 30 loops each)")


# ```python
# %timeit -n 30 s_cat.nunique()
# ```

# In[16]:


print("403 µs ± 22.5 µs per loop (mean ± std. dev. of 7 runs, 30 loops each)")


# - 2

# In[17]:


df.cut = df.cut.astype('category').cat.reorder_categories([
        'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],ordered=True)
df.clarity = df.clarity.astype('category').cat.reorder_categories([
        'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],ordered=True)
res = df.sort_values(['cut', 'clarity'], ascending=[False, True])


# In[18]:


res.head(3)


# In[19]:


res.tail(3)


# - 3

# In[20]:


df.cut = df.cut.cat.reorder_categories(df.cut.cat.categories[::-1])
df.clarity = df.clarity.cat.reorder_categories(df.clarity.cat.categories[::-1])
# 方法一：利用cat.codes
df.cut = df.cut.cat.codes
clarity_cat = df.clarity.cat.categories
# 方法二：使用replace映射
df.clarity = df.clarity.replace(dict(zip(clarity_cat, np.arange(len(clarity_cat)))))
df.head(3)


# - 4

# In[21]:


q = [0, 0.2, 0.4, 0.6, 0.8, 1]
point = [-np.infty, 1000, 3500, 5500, 18000, np.infty]
avg = df.price / df.carat
df['avg_cut'] = pd.cut(avg, bins=point, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
df['avg_qcut'] = pd.qcut(avg, q=q, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
df.head()


# - 5

# In[22]:


df.avg_cut.unique()


# In[23]:


df.avg_cut.cat.categories


# In[24]:


df.avg_cut = df.avg_cut.cat.remove_categories(['Very Low', 'Very High'])
df.avg_cut.head(3)


# - 6

# In[25]:


interval_avg = pd.IntervalIndex(pd.qcut(avg, q=q))
interval_avg.right.to_series().reset_index(drop=True).head(3)


# In[26]:


interval_avg.left.to_series().reset_index(drop=True).head(3)


# In[27]:


interval_avg.length.to_series().reset_index(drop=True).head(3)


# ## 三、有序类别下的逻辑斯蒂回归
# 
# 
# 逻辑斯蒂回归是经典的分类模型，它将无序的类别作为目标值来进行模型的参数训练。对于有序类别的目标值，虽然我们仍然可以将其作为无序类别来输入模型，但这样做显然损失了类别之间的顺序关系，而有序类别下的逻辑斯蒂回归（Ordinal Logistic Regression）就能够对此类问题进行处理。
# 
# 
# 设样本数据为$X=(x_1,...,x_n),x_i\in R^d(1\leq i \leq n)$，$d$是数据特征的维度，标签值为$y=(y_1,...,y_n), y_i\in \{C_1,...,C_k\}(1\leq i\leq n)$，$C_i$是有序类别，$k$是有序类别的数量，记$P(y\leq C_0\vert x_i)=0(1\leq i\leq n)$。设参数$w=(w_1,...,w_d),\theta=(\theta_0,\theta_1,...,\theta_k)$，其中$\theta_0=-\infty,\theta_k=\infty$，则OLR模型为：
# 
# $$P(y_i\leq C_j\vert x_i)=\frac{1}{1+\exp^{-(\theta_j-w^Tx_i)}}, 1\leq i \leq n, 1\leq j \leq k$$
# 
# 由此可得已知$x_i$的情况下，$y_i$取$C_j$的概率为：
# 
# $$P(y_i=C_j\vert x_i) = P(y_i\leq C_j\vert x_i)-P(y_i\leq C_{j-1}\vert x_i), 1\leq j\leq k$$
# 
# 
# 从而对数似然方程为：
# 
# $$\log L(w,\theta\vert X,y)=\sum_{i=1}^n\sum_{j=1}^k \mathbb{I}_{\{y_i=C_j\}}\log [\frac{1}{1+\exp^{-(\theta_j-w^Tx_i)}}-\frac{1}{1+\exp^{-(\theta_{j-1}-w^Tx_i)}}]$$
# 
# mord包能够对上述OLR模型的参数$w$和参数$\theta$进行求解，可以使用conda install -c conda-forge mord命令下载。
# 
# 首先，我们读取1个目标值为4个有序类别的数据集：

# In[28]:


df = pd.read_csv("data/ch9/olr.csv")
df.head()


# 已知y中元素的大小关系为“Bad”$\leq$“Fair”$\leq$“Good”$\leq$“Marvellous”，此时先将y转换至有序类别编号：

# In[29]:


df.y = df.y.astype("category").cat.reorder_categories(['Bad', 'Fair',
    'Good', 'Marvellous'], ordered=True).cat.codes
df.y.head()


# 从mord中导入LogisticAT进行参数训练：

# In[30]:


from mord import LogisticAT
clf = LogisticAT()
X = df.iloc[:,:4]
clf.fit(X, df.y)


# 此时，我们就能通过clf.coef_和clf.theta_分别访问$w$和$\theta$：

# In[31]:


clf.coef_ # w1, w2, w3, w4


# In[32]:


clf.theta_ # 每一个类别的theta_j


# 从而就能通过predict_proba()方法对每一个样本的特征$x_i$计算$y_i$属于各个类别的概率，取$\arg\max$后即得到输出的类别。

# In[33]:


res = clf.predict_proba(X).argmax(1)[:5] # 取前五个
res


# - 现有1个新的样本点$x_{new}=[-0.5, 0.3, 0.4, 0.1]$需要进行预测，请问它的predict_proba()概率预测结果是如何得到的？请写出计算过程。

# In[34]:


x_new = np.array([[-0.5, 0.3, 0.4, 0.1]])
clf.predict_proba(x_new)


# - 数据集data/ch9/car.csv中，含有6个与汽车有关的变量：“buying”、“maint”、“doors”、“persons”、“lug_boot”和“safety”分别指购买价格、保养价格、车门数量、车载人数、车身大小和安全等级，需要对汽车满意度指标“accpet”进行建模。汽车满意度是有序类别变量，由低到高可分为“unacc”、“acc”、“good”和“vgood”4个类别。请利用有序类别的逻辑斯蒂回归，利用6个有关变量来构建关于汽车满意度的分类模型。

# In[35]:


df = pd.read_csv("data/ch9/car.csv")
df.head()


# ```text
# 【解答】
# ```
# 
# - 1

# In[36]:


df = pd.read_csv("data/ch9/olr.csv")
df.y = df.y.astype("category").cat.reorder_categories(['Bad', 'Fair',
    'Good', 'Marvellous'], ordered=True).cat.codes
from mord import LogisticAT
clf = LogisticAT()
X = df.iloc[:,:4]
clf.fit(X, df.y)


# In[37]:


x = x_new[0]
theta_ = np.r_[-np.inf, clf.theta_, np.inf]
p = 1 / (1 + np.exp(-theta_+x.dot(clf.coef_)))
p = np.diff(p)
p


# - 2

# In[38]:


df = pd.read_csv("data/ch9/car.csv")
X = pd.get_dummies(df.iloc[:, :-1])
y = df.accept.astype("category").cat.reorder_categories(
    ["unacc", "acc", "good", "vgood"], ordered=True).cat.codes
clf = LogisticAT()
clf.fit(X, y)

