
# coding: utf-8

# In[19]:

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

filename = sys.argv[1]
col_0_name = "numPairs"
y_label_0 = "Insertion Throughput (billions pairs/sec)"
x_label_0 = "Load Factor"
title_0 = "V100 GPU Insertions, 900M Pairs, Multi-Pass Gather"
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "V100 GPU Querying, 900M Pairs, Multi-Pass Gather"
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

df['hfInsert'] = 900000000/df.hfInsert
df['cuInsert'] = 900000000/df.cuInsert
df['thSort'] = 900000000/df.thSort
ax1.plot(df['numPairs'], df['hfInsert'], label='HashFight', marker='o')
ax1.plot(df['numPairs'], df['cuInsert'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df['numPairs'], df['thSort'], label='Thrust Sort', marker='x')
ax1.legend(loc=4)
#ax1.legend(loc='upper right', bbox_to_anchor=(1.88,1.62))
#ax1.legend(frameon=False)
#ax1.set_xticks(np.arange(1.03,2.0))
plt.ylabel(y_label_0)
plt.xlabel(x_label_0)
#plt.axis([50, 500, 0, 2000])
#plt.xticks(np.arange(1.0,2.1))
#plt.yticks(np.arange(0,7,1))
#plt.margins(x=0,y=0)
plt.title(title_0)
#plt.yscale('log')

# In[29]:


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)

df['hfQuery'] = 900000000/df.hfQuery
df['cuQuery'] = 900000000/df.cuQuery
df['thSearch'] = 900000000/df.thSearch
ax1.plot(df['numPairs'], df['hfQuery'], label='HashFight', marker='o')
ax1.plot(df['numPairs'], df['cuQuery'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df['numPairs'], df['thSearch'], label='Thrust Search', marker='x')
ax1.legend(loc=1)
plt.ylabel(y_label_1)
plt.xlabel(x_label_1)
#plt.axis([50, 1450, 0, 600000000])
plt.title(title_1)
#plt.yscale('log')

#plt.tight_layout()
plt.show()

# In[13]:


df

