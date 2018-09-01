
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv('103InsertTimes.csv',header=None)


# In[17]:


fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(df[:,0], df[:,1], label='insert_time_hashfight')
ax1.plot(df[:,0], df[:,2], label='query_time_hashfight')
ax1.plot(df[:,0], df[:,3], label='insert_time_cuckoo')
ax1.plot(df[:,0], df[:,4], label='query_time_cuckoo')
ax1.legend(loc=2)
plt.show()

# In[13]:


#df

