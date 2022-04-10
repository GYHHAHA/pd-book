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


# # 第一章
# 
# ## 零、练一练
# 
# ```{admonition} 练一练
# 给定一个包含5个英语单词的列表，请构造1个字典以列表的元素为键，以每个键对应的单词字母个数为值。
# ```

# In[2]:


en_list = ["apple", "banana", "peach", "pineapple", "watermelon"]


# In[3]:


en_dict = {word: len(word) for word in en_list}
en_dict


# ```{admonition} 练一练
# 给定3个二维整数列表$L_1$, $L_2$, $L_3$，它们的形状都是$30\times20$，即每个列表中包含30个内层列表，并且每一个内层列表中包含20个整数。请利用列表推导式，构造一个形状相同的新列表 $L_{new}$，其满足任意一个位置的值是$L_1$, $L_2$, $L_3$相应位置的最小值。
# ```

# In[4]:


import random
random.seed(0)
n = 10000
L1 = [[random.randint(-n, n) for j in range(20)] for i in range(30)]
L2 = [[random.randint(-n, n) for j in range(20)] for i in range(30)]
L3 = [[random.randint(-n, n) for j in range(20)] for i in range(30)]


# In[5]:


L_new = [[min(L1[i][j], L2[i][j], L3[i][j]) for j in range(20)] for i in range(30)]


# ```{admonition} 练一练
# 对于上面构造的my_list，请选出包含3的整数倍的内层列表。
# ```

# In[6]:


my_list = [[1, 2], [3, 4, 5], [6], [7, 8], [9]]


# In[7]:


list(filter(lambda x: any(i%3==0 for i in x), my_list))


# ``` {admonition} 练一练
# 请用zip函数完成上述例子中enumerate的功能。
# ```

# In[8]:


L2 = ["apple", "ball", "cat", "dog", "eye"]


# In[9]:


for index, value in zip(range(len(L2)), L2):
    print(index, value)


# ```{admonition} 练一练
# split()函数从功能上更类似于concatenate()的逆操作还是stack()的逆操作？请说明理由。
# ```
# 
# split()函数更类似于concatenate()的逆操作，首先stack必须由多个尺寸相同的数组来拼接，新产生的维度大小取决于拼接数组的数量，而concatenate的被拼接数组在拼接维度上可以不一致且不会产生新维度。下面的例子更清楚地反映了这组互逆操作的特性：

# In[10]:


arr = np.random.rand(10, 20, 30)
arr_new = np.concatenate(np.split(arr, indices_or_sections=5, axis=1), axis=1)


# In[11]:


(arr == arr_new).all()


# ```{admonition} 练一练
# 请使用repeat函数分别构造两个$10\times 10$的数组，第一个数组要求第i行的元素值都为i，第二个数组要求第i列的元素值都为i。
# ```

# In[12]:


arr = np.arange(1, 11)


# In[13]:


rep_1 = np.repeat(arr[:, None], 10, -1)


# In[14]:


rep_2 = np.repeat(arr[None, :], 10, 0)


# ```{admonition} 练一练
# 与Python中字符串的切片类似，numpy数组切片的首末端点以及步长都可以是负数，例如arr是一个大小为$10\times 5$的数组，那么arr[-2:-10:-3, 1:-1:2]切片结果的大小为$3\times 2$。请给出一些相应的例子，并观察结果是否与预期一致。
# ```

# In[15]:


arr = np.random.rand(10, 5)


# In[16]:


arr[-2:-10:-3, 1:-1:2]


# -2:-10:-3表示从倒数第2行开始切片，逆向地每3个取一次，直到倒数第10行且不包含倒数第10行，因此即为倒数第8行、倒数第5行和倒数第2行共计三行。列上的切片类似，从第一列开始切片，每两列取一次，直到倒数第一列且不包含倒数第一列，因此即为第2列、第4列共计两列。
# 
# ```{admonition} 练一练
# 对于如下的数组维度组合，判断使用逐元素运算是否会报错，如果不会请直接写出广播结果的维度：
# - $1\times 3\times 5$和$3\times 1$
# - $3\times 5\times 3\times 4$和$1\times 3\times 1$
# - $3\times 2\times 1\times 5$和$2\times 5$
# ```
# 
# - $1\times 3\times 5$
# - $3\times 5\times 3\times 4$
# - $3\times 2\times 2\times 5$
# 
# ```{admonition} 练一练
# 对于上述price维度的修改，除了使用np.newaxis之外，还可以使用reshape()和expand_dims()来实现，请分别使用这两种方法完成等价操作。
# ```

# In[17]:


price = np.array([25,20,30]) # 假设给定的单价是25、20和30


# In[18]:


price[:, None] # newaxis即None


# In[19]:


price.reshape(-1, 1)


# In[20]:


np.expand_dims(price, -1)


# ```{admonition} 练一练
# 仿照上面的例子，给出按年级统计学生总人数的方案，即返回数组的包含3个元素，分别为各年级中所有学校和班级的学生人数之和。
# ```

# In[21]:


my_matrix = np.random.randint(20, 40, 24).reshape(2, 3, 4)
my_matrix.sum((0, 2)).shape


# ```{admonition} 练一练
# Softmax函数在深度学习的模型设计中有重要应用，对于1维数组$[x_1,...,x_n]$进行Softmax归一化时，每一个元素被修正为$\tilde{x}_i={\rm Softmax(x_i)}=\frac{\exp(x_i)}{\sum_{i=1}^n\exp(x_i)}$。现给定一个二维数组，请对其进行逐行Softmax归一化，且不得使用for循环。
# ```

# In[22]:


arr = np.random.rand(4, 4)
exp_arr = np.exp(arr)
res = exp_arr / exp_arr.sum(1)[:, None]
res


# ```{admonition} 练一练
# 阅读逻辑函数的相关内容，完成下列练习：
# - 逻辑运算符的优先顺序是怎样的？其左右的数组能够被广播吗？请构造例子说明。
# - 给定一个维度为$m\times n$的整数数组，请返回一个元素全为0或1的同维度数组，且满足元素取1当且仅当该位置在原数组中的对应元素是原数组中同行元素的最大值。
# ```
# 
# - 优先级从高到低：not("~")、and("&")、or(“|”)，可以广播

# In[23]:


a = np.array([True])
b = np.array([False])


# In[24]:


~a | a # 说明not优先级高于or


# In[25]:


~b & b # 说明not优先级高于and


# In[26]:


a | b & b # 说明and优先级高于or


# In[27]:


a = np.array([True, False])
b = np.array([True, False])
a | b[:, None] # 广播


# - 方案如下

# In[28]:


a = np.random.randint(0, 100, (5, 5))


# In[29]:


np.where(a == a.max(1)[:, None], 1, 0)


# ```{admonition} 练一练
# 利用round函数将上例中的随机矩阵按第1位小数四舍五入取整，依次筛选出矩阵中满足如下条件的行：
# - 行元素至多有一个1
# - 行元素至少有一个0
# - 行元素既非全0又非全1
# ```

# In[30]:


my_array = np.random.rand(1000, 3)
arr = my_array.round()
arr_1 = arr[arr.sum(1) <= 1]
arr_2 = arr[~arr.all(1)]
arr_3 = arr[~arr.all(1) & ~(1-arr).all(1)]


# ```{admonition} 练一练
# np.clip(array, min, max)是一种截断函数，对于数组中超过max的值会被截断为max，数组中不足min的值会被截断为min。请用np.where()实现这个函数。
# ```

# In[31]:


arr = np.array([1, 2, 3, 4, 5])


# In[32]:


np.clip(arr, 2, 4)


# In[33]:


res = np.where(arr<=4, arr, 4)
res = np.where(arr>=2, res, 2)
res


# ```{admonition} 练一练
# 在上面这个例子中，nonzero()的输入a是1维数组，通过a[np.nonzero(a)]能够取出数组中所有的非零元素值。事实上，nonzero()函数也能够以高维数组作为参数传入，此时其返回值代表了什么含义？a[np.nonzero(a)]仍然能够选出数组中所有的非零元素值吗？请解释理由。
# ```
# 
# 通过查阅文档可知，返回值分别代表了所有非零元素在每一个维度上对应的索引，在1.2.3中我们曾给出了一个通过在相应位置传入同长度列表来索引对应位置元素的例子（见“target[[0, 1], [0, 1], [0, 1]]”），这里的做法是完全一致的：

# In[34]:


a = np.array([[0,1],[0,2]])
# x为所有非零元素在dim=0上的索引，y为所有元素在dim=1上的索引
x, y = np.nonzero(a)
a[x, y]


# ## 一、利用列表推导式实现矩阵乘法
# 
# 记矩阵$A_{m\times n}$，矩阵$B_{n\times p}$，记$A$与$B$的矩阵乘法结果为矩阵$C_{m\times p}$，此时其第$i$行第$j$列的元素满足
# 
# $$
# C_{ij}=\sum_{k=1}^nA_{ik}B_{kj}
# $$
# 
# 在numpy中可以使用“@”符号来进行矩阵乘法：

# In[35]:


A = np.arange(6).reshape(2, -1)
B = np.arange(6).reshape(3, -1)
A @ B


# 请利用列表推导式来实现矩阵乘法。
# 
# ```text
# 【解答】
# ```

# In[36]:


res = [
    [
        sum(
                A[i][k] * B[k][j]
                for k in range(A.shape[1])
        )
        for j in range(B.shape[1])
    ] for i in range(A.shape[0])
]
res


# ## 二、计算卡方统计量
# 
# 设矩阵$A_{m\times n}$，记$B_{ij} = \frac{(\sum_{i=1}^mA_{ij})\times (\sum_{j=1}^nA_{ij})}{\sum_{i=1}^m\sum_{j=1}^nA_{ij}}$，定义矩阵$A$对应的卡方统计量如下
# 
# $$
# \chi^2 = \sum_{i=1}^m\sum_{j=1}^n\frac{(A_{ij}-B_{ij})^2}{B_{ij}}
# $$
# 
# 
# 请利用numpy对如下构造的矩阵$A$计算相应的卡方统计量$\chi^2$。

# In[37]:


np.random.seed(0)
A = np.random.randint(10, 20, (8, 5))


# ```text
# 【解答】
# ```

# In[38]:


B = A.sum(0) * A.sum(1)[:, None] / A.sum()
res = ((A - B) ** 2 / B).sum()
res


# ## 三、统计某商店的月度销量情况
# 
# 在文件夹data/ch1/shop_sales下存放了200类货品从2001年1月至2020年12月的月度销量数组，每个数组的大小为$Y\times M$，其中$Y=20$表示年维度，$M=12$表示月维度。

# In[39]:


# 使用np.load能够加载npy数组
# 使用np.save("文件路径/my_arr.npy")能够将数组保存到本地
arr = np.load("data/ch1/shop_sales/product_1.npy")
arr.shape


# - 计算各季度（从1月至12月，每3个月表示一个季度）的销售总量，输出结果为$Y\times Q$的数组，其中$Q=4$表示季度维度。
# - 计算各月不同种类货品销量的方差，输出结果为$Y\times M$的数组。
# - 在文件data/ch1/increase_rate.npy中记录了各类货品每月关于上月的单价涨幅，数组大小为$200\times 20*12$，其中$20*12$表示20年且每年12个月，共计240个月。在文件data/ch1/unit_price.npy中记录了各类货品在2015年1月的单价，数组长度为$200$。请计算所有货品从2001年1月至2020年12月的单价，输出结果为$Y\times M\times 200$。
# - 结合上一小问的结果，求出各类货品最大月度销售额的所在月份，输出结果为长度为$200$的1维数组，其中每个元素为相应月份的字符串表示，例如“2008-05”。
# 
# ```text
# 【解答】
# ```
# 
# - 1

# In[40]:


path = "data/ch1/shop_sales/product_%d.npy"
arr = np.stack([np.load(path%i) for i in range(1, 201)], axis=0)
res_1 = arr.reshape(200, 20, 4, -1).sum((0, -1))


# - 2

# In[41]:


res_2 = arr.var(0)
res_2.shape


# - 3

# In[42]:


rate = np.load("data/ch1/increase_rate.npy")
base = np.load("data/ch1/unit_price.npy")
price = np.empty((200, 20*12))
n = 14 * 12 # 2015年对应的索引位置
price[:, n] = base
price[:, n+1:] = (rate[:, n+1:] + 1).cumprod(-1) * base[:, None]
price[:, :n] = 1/(rate[:, 1:n+1] + 1)[:, ::-1].cumprod(-1)[:, ::-1] * base[:, None]
price = price.reshape(200, 20, 12).transpose(1, 2, 0)
price.shape


# - 4

# In[43]:


idx = (arr.transpose(1, 2, 0) * price).reshape(-1, 200).argmax(0)
year, month = idx // 12 + 2001, idx % 12 + 1
L = ["%d-%02d"%(y, m) for y, m in zip(year, month)]
L[:5] # 展示前5个

