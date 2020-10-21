
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
x_label_0 = "Load Factor"
title_0 = "K40 and V100 GPU Insertions, Variable Load Factor"
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "K40 and V100 GPU Querying, Variable Load Factor"
minX_0 = 50 
maxX_0 = 500
minY_0 = 0
maxY_0 = 2000
minX_1 = 50 
maxX_1 = 500
minY_1 = 0
maxY_1 = 2000


df = pd.read_csv(filename, header=None, names=['numPairs', 'hfInsert', 'hfQuery', 'cuInsert', 'cuQuery', 'thSort', 'thSearch', 'hfInsert2', 'hfQuery2', 'cuInsert2', 'cuQuery2', 'thSort2', 'thSearch2'])


# In[28]:


fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

df['hfInsert'] = 900000000/df.hfInsert
df['cuInsert'] = 900000000/df.cuInsert
df['thSort'] = 900000000/df.thSort
df['hfInsert2'] = 300000000/df.hfInsert2
df['cuInsert2'] = 300000000/df.cuInsert2
df['thSort2'] = 300000000/df.thSort2
ax1.plot(df['numPairs'], df['hfInsert'], label='HashFight, V100, 900M', marker='o')
ax1.plot(df['numPairs'], df['hfInsert2'], label='HashFight, K40, 300M', marker='o')
ax1.plot(df['numPairs'], df['cuInsert'], label='CUDPP, V100, 900M', linestyle='--')
ax1.plot(df['numPairs'], df['cuInsert2'], label='CUDPP, K40, 300M', linestyle='--')
ax1.plot(df['numPairs'], df['thSort'], label='Thrust Sort, V100, 900M', marker='x')
ax1.plot(df['numPairs'], df['thSort2'], label='Thrust Sort, K40, 300M', marker='x')
#ax1.legend(loc=0)
plt.ylabel(y_label_0)
plt.xlabel(x_label_0)
#plt.axis([50, 500, 0, 2000])
plt.title(title_0)
#plt.yscale('log')

# In[29]:


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)

df['hfQuery'] = 900000000/df.hfQuery
df['cuQuery'] = 900000000/df.cuQuery
df['thSearch'] = 900000000/df.thSearch
df['hfQuery2'] = 300000000/df.hfQuery2
df['cuQuery2'] = 300000000/df.cuQuery2
df['thSearch2'] = 300000000/df.thSearch2
ax1.plot(df['numPairs'], df['hfQuery'], label='HashFight, V100, 900M', marker='o')
ax1.plot(df['numPairs'], df['hfQuery2'], label='HashFight, K40, 300M', marker='o')
ax1.plot(df['numPairs'], df['cuQuery'], label='CUDPP, V100, 900M', linestyle='--')
ax1.plot(df['numPairs'], df['cuQuery2'], label='CUDPP, K40, 300M', linestyle='--')
ax1.plot(df['numPairs'], df['thSearch'], label='Thrust Search, V100, 900M', marker='x')
ax1.plot(df['numPairs'], df['thSearch2'], label='Thrust Search, K40, 300M', marker='x')
ax1.legend(loc=0)
plt.ylabel(y_label_1)
plt.xlabel(x_label_1)
#plt.axis([50000000, 500000000, 0, 600000000])
plt.title(title_1)
#plt.yscale('log')

plt.show()

# In[13]:


df

