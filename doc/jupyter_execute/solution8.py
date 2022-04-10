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


# # 第八章
# 
# ## 零、练一练
# 
# ```{admonition} 练一练
# 对于序列a=pd.Series([[1,2], [4,5]])，转为string类型后使用str切片，即a.astype("string").str[::-1]，与直接使用str切片，即a.str[::-1]，它们的结果分别是什么？请进行相应的解释。
# ```

# In[2]:


a = pd.Series([[1,2], [4,5]])


# 当转为字符串后，每个单元格的元素都会更换为元素调用__str__()返回的字符串对象，此时切片是逐个字符串元素的切片：

# In[3]:


a.iloc[0].__str__()


# In[4]:


a.astype("string").str[::-1]


# 当直接使用切片时，等价于直接对每个元素进行[::-1]操作，由于内部存储的时列表，因此每个对应位置返回了列表的反转：

# In[5]:


a.str[::-1]


# ````{admonition} 练一练
# 使用如下的语句可从my_file.txt中读取文本：
# ```python
# with open("data/ch8/my_file.txt", "w") as f:
#     text = f.read()
#     # 进行后续文本操作
# ```
# 使用如下的语句可将文本保存到my_file.txt中：
# ```python
# text = "aaa\nbbb"
# with open("my_file.txt", "w") as f:
#     f.write(text)
# ```
# 请结合正则表达式相关的知识，完成以下内容：
# - 读取data/ch8/regex1.txt中的数据（文本均随机生成），删除“#begin”和“#close”所在行及其中间的所有行，处理后保存至data/ch8/regex1_result.txt中。
# - 读取data/ch8/regex2.txt中的数据，将“\section{×××}”、“\subsection{×××}”和“\subsubsection{×××}”分别替换为“# ×××”、“## ×××”和“### ×××”，其中“×××”指代花括号中的标题名，处理后保存至data/ch8/regex2_result.txt中。
# ````
# 
# - 1

# In[6]:


import re
with open("data/ch8/regex1.txt", "r") as f:
    text = f.read()
    res = re.sub("#begin[\S\s]*?#close", "", text)
with open("data/ch8/regex1_result.txt", "w") as f:
    f.write(res)


# - 2

# In[7]:


with open("data/ch8/regex2.txt", "r", encoding="utf-8") as f:
    text = f.read()
    res = re.sub(r"\\((?:sub)*)section\{([\s\S]+?)\}", r"sub\1 \2", text)
    res = re.sub(r"sub", r"#", res)
with open("data/ch8/regex2_result.txt", "w") as f:
    f.write(res)


# ```{admonition} 练一练
# 在上述的两个负向断言例子中，如果把“[件|\d]”和“[少|\d]”分别修改为“件”和“少”，此时匹配结果如何变化？请解释原因。
# ```
# 
# 在第一个例子中，想要匹配的是“元”前面的那个数字，不记录“件”前面的那个数字，但如果此时不剔除\d，那么此时50这个数字符合\d\d，能够匹配(\d+)(?!件)，因此5出现在了结果中。在第二个例子中，造成0匹配的原因类似，50恰好能够匹配(?<!少)(\d+)，因此返回的结果中包括0.
# 
# ## 一、房屋数据的文本提取
# 
# 现有一份房屋信息数据集如下：

# In[8]:


df = pd.read_csv('data/ch8/house.csv')
df.head(3)


# - 将year列改为整数年份存储。
# - 将floor列替换为Level、Highest两列，其中的元素分别为string类型的层类别（高层、中层、低层）与整数类型的最高层数。
# - 计算每平米均价avg_price，以××元/平米的格式存储到表中，其中××为整数。
# 
# ```text
# 【解答】
# ```

# In[9]:


df = pd.read_csv('data/ch8/house.csv')
df.year = pd.to_numeric(df.year.str[:-2]).astype('Int64')
df.head()


# In[10]:


pat = '(\w层)（共(\d+)层）'
new_cols = df.floor.str.extract(pat).rename(columns={0:'Level', 1:'Highest'})
df = pd.concat([df.drop(columns=['floor']), new_cols], axis=1)
df.head()


# In[11]:


s_area = pd.to_numeric(df.area.str[:-1])
s_price = pd.to_numeric(df.price.str[:-1])
df['avg_price'] = ((s_price/s_area)*10000).astype('int').astype('string') + '元/平米'
df.head()


# ## 二、巴洛克作曲家的年龄统计
# 
# 
# 巴洛克时期是西方音乐的发展过程中的重要时期，它上承文艺复兴时期，下启古典主义时期，期间诞生了许多伟大的作曲家。在data/ex-ch8-2-baroque.txt中存放了巴洛克作曲家（含部分文艺复兴晚期作曲家）的名字和生卒年份：

# In[12]:


df = pd.read_table("data/ch8/baroque.txt")
df.head()


# - 请筛选出能够确定出生与去世年份的作曲家，并提取他们的姓名、出生年和去世年。
# - 约翰.塞巴斯蒂安.巴赫（Johann Sebastian Bach）是重要的巴洛克作曲家，请问在数据表中寿命超过他的作曲家比例为多少？
# 
# ```text
# 【解答】
# ```
# 
# - 1

# In[13]:


df = pd.read_table("data/ch8/baroque.txt")
pat = '(?P<Name>[\w\s]+)\s\((?P<birth>\d{4})-(?P<death>\d{4})\)'
res = df.iloc[:, 0].str.extract(pat).dropna().reset_index(drop=True)
res.head()


# - 2

# In[14]:


res.birth = res.birth.astype("int")
res.death = res.death.astype("int")
bach = res.query("Name=='Johann Sebastian Bach'").iloc[0]
bach_age = bach.death - bach.birth
((res.death - res.birth) > bach_age).mean()


# ## 三、汇总显卡测试的结果
# 
# 在data/ch8/benchmark.txt文件中记录了RTX3090显卡某次性能测评的日志结果，每一条日志有如下结构：
# 
# ```text
# Benchmarking #2# #4# precision type #1#
# #1#  model average #2# time :  #3# ms
# ```
# 
# 其中#1#代表的是模型名称，#2#的值为train(ing)或inference，表示训练状态或推断状态，#3#表示耗时，#4#表示精度，其中包含了float、half、double这3种类型，下面是一个具体的例子：
# 
# ```text
# Benchmarking Inference float precision type resnet50
# resnet50  model average inference time :  13.426570892333984 ms
# ```
# 
# 请把日志结果进行整理，变换成如下状态，行索引用相应模型名称填充，按照字母顺序排序，数值保留3位小数：

# In[15]:


df = pd.read_table('data/ch8/benchmark.txt').iloc[9:-2].reset_index(drop=True)
res1 = df.loc[0::2,'start'].str.extract('Benchmarking (?P<One>Training|Inference) (?P<Two>float|half|double) precision type (?P<Three>\w+)').reset_index(drop=True)
res2 = pd.to_numeric(df.loc[1::2,'start'].str.extract('.+time :  (?P<Time>\d+\.\d{3}).+').Time).reset_index(drop=True)
res = pd.concat([res1.One +'_'+ res1.Two, res1.Three, res2],axis=1).set_index([0,'Three']).unstack('Three').droplevel(0, axis=1).T
res = res.reindex(pd.MultiIndex.from_product([['Training','Inference'],['half','float','double']]).map(lambda x:x[0]+'_'+x[1]),axis=1)
res.head()


# ```text
# 【解答】
# ```

# In[16]:


df = pd.read_table('data/ch8/benchmark.txt').iloc[9:-2].reset_index(drop=True)
pat1 = 'Benchmarking (?P<One>Training|Inference) (?P<Two>float|half|double) precision type (?P<Three>\w+)'
pat2 = '.+time :  (?P<Time>\d+\.\d{3}).+'
res1 = df.loc[0::2,'start'].str.extract(pat1).reset_index(drop=True)
res2 = pd.to_numeric(df.loc[1::2,'start'].str.extract(pat2).Time).reset_index(drop=True)
res = pd.concat([res1.One +'_'+ res1.Two, res1.Three, res2],axis=1).set_index([0,'Three'])
res = res.unstack('Three').droplevel(0, axis=1).T
idx = pd.MultiIndex.from_product([['Training','Inference'],['half','float','double']])
idx = idx.map(lambda x:x[0]+'_'+x[1])
res = res.reindex(idx ,axis=1)
res.head()

