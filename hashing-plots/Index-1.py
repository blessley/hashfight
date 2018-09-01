
# coding: utf-8

# In[19]:

import sys
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

filename = sys.argv[1]

df = pd.read_csv(filename, header=None, names=['num_pairs', 'insert_time_hashfight', 'query_time_hashfight', 'insert_time_cuckoo', 'query_time_cuckoo'])


# In[28]:


fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(df['num_pairs'], df['insert_time_hashfight'], label='VTKm-HashFight', marker='o')
ax1.plot(df['num_pairs'], df['insert_time_cuckoo'], label='CUDPP-Cuckoo', linestyle='--')
ax1.legend(loc=2)
plt.ylabel('Insertion Time (ms)')
plt.xlabel('Load Factor')
plt.axis([1.0, 2.0, 0, 2100])
plt.title('Insertion Time per Load Factor, 225 million pairs')
#plt.yscale('log')

# In[29]:


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)

ax1.plot(df['num_pairs'], df['query_time_hashfight'], label='VTKm-HashFight', marker='o')
ax1.plot(df['num_pairs'], df['query_time_cuckoo'], label='CUDPP-Cuckoo', linestyle='--')
ax1.legend(loc=2)
plt.ylabel('Query Time (ms)')
plt.xlabel('# pairs (millions)')
plt.axis([1.0, 2.0, 0, 2100])
plt.title('Query Time per Load Factor, 225 million pairs')
#plt.yscale('log')

plt.show()

# In[13]:


df

