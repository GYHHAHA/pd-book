#!/usr/bin/env python
# coding: utf-8

# In[18]:


df = pd.read_csv('data/learn_pandas.csv')
res = df.loc[(df.School.isin(['A', 'B'])) & (df.Gender == 'Female') & (df.Grade == 'Freshman')]
res.head()


# In[17]:


df = pd.read_csv('data/learn_pandas.csv')
res = df.query("(School in ['A', 'B']) and"
               "(Gender == 'Female') and"
               "(Grade == 'Freshman')")
res.head()

