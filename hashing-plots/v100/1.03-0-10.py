
# coding: utf-8

# In[19]:

import sys
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

filename = sys.argv[1]
col_0_name = "numPairs"
y_label_0 = "Insertion Throughput (billions pairs/sec)"
x_label_0 = "Billions of Key-Value pairs"
title_0 = "V100 GPU Insertions, 1.03 Load Factor"
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Billions of Key-Value Pairs"
title_1 = "V100 GPU Querying, 1.03 Load Factor"
minX_0 = 50 
maxX_0 = 500
minY_0 = 0
maxY_0 = 2000
minX_1 = 50 
maxX_1 = 500
minY_1 = 0
maxY_1 = 2000


df = pd.read_csv(filename, header=None, names=['numPairs', 'hfInsert', 'hfQuery', 'cuInsert', 'cuQuery', 'thSort', 'thSearch'])


# In[28]:


fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

df['hfInsert'] = df.numPairs/df.hfInsert
df['cuInsert'] = df.numPairs/df.cuInsert
df['thSort'] = df.numPairs/df.thSort
ax1.plot(df['numPairs'], df['hfInsert'], label='HashFight', marker='o')
ax1.plot(df['numPairs'], df['cuInsert'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df['numPairs'], df['thSort'], label='Thrust Sort', marker='x')
ax1.legend(loc=7)
plt.ylabel(y_label_0)
plt.xlabel(x_label_0)
#plt.axis([50, 500, 0, 2000])
plt.title(title_0)
#plt.yscale('log')

# In[29]:


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)

df['hfQuery'] = df.numPairs/df.hfQuery
df['cuQuery'] = df.numPairs/df.cuQuery
df['thSearch'] = df.numPairs/df.thSearch
ax1.plot(df['numPairs'], df['hfQuery'], label='HashFight', marker='o')
ax1.plot(df['numPairs'], df['cuQuery'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df['numPairs'], df['thSearch'], label='Thrust Search', marker='x')
ax1.legend(loc=7)
plt.ylabel(y_label_1)
plt.xlabel(x_label_1)
#plt.axis([50, 1450, 0, 600000000])
plt.title(title_1)
#plt.yscale('log')

plt.show()

# In[13]:


df

