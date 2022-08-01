#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.DataFrame([[1, 2], [3, 4]], index=["row_1", "row_2"], columns=["col_1", "col_2"])
# rolling across axis=0 works fine:
df.rolling(window=2, axis=0, min_periods=1).aggregate([np.sum, np.mean])


# In[2]:


get_ipython().run_line_magic('load_ext', 'cython')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
s = pd.Series(np.random.choice(list("ABCD"), size=100, p=[0.1,0.2,0.5,0.2]))
data = s.value_counts().sort_index()
_ = plt.pie(data.values,labels=data.index)


# In[4]:


fig, ax = plt.subplots(figsize=(8,6)) #figsize指图片尺寸
ax.plot([1,2,3], [1,2,3])


# In[5]:


get_ipython().run_cell_magic('cython', '', '\nfrom cython.parallel import prange\n\ndef test():\n    cdef int i, j, s\n    for i in prange(5, nogil=True):\n        j = 0\n        while j < 10:\n            j = j + 1\n        s += j\n    return s')


# In[6]:


test()

