
# coding: utf-8

# In[19]:

import sys
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

filename = sys.argv[1]
col_0_name = "numPairs"
y_label_0 = "Insertion Throughput (millions pairs/sec)"
x_label_0 = "Load Factor"
title_0 = "K40 GPU Insertions, 300M Pairs"
y_label_1 = "Query Throughput (millions pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "K40 GPU Querying, 300M Pairs"
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

df['hfInsert'] = 300000000/df.hfInsert
df['cuInsert'] = 300000000/df.cuInsert
df['thSort'] = 300000000/df.thSort
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

df['hfQuery'] = 300000000/df.hfQuery
df['cuQuery'] = 300000000/df.cuQuery
df['thSearch'] = 300000000/df.thSearch
ax1.plot(df['numPairs'], df['hfQuery'], label='HashFight', marker='o')
ax1.plot(df['numPairs'], df['cuQuery'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df['numPairs'], df['thSearch'], label='Thrust Search', marker='x')
ax1.legend(loc=1)
plt.ylabel(y_label_1)
plt.xlabel(x_label_1)
#plt.margins(x=0)
#plt.axis([50000000, 500000000, 0, 600000000])
plt.title(title_1)
#plt.yscale('log')

plt.show()

# In[13]:


df

