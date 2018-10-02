
# coding: utf-8

# In[19]:

import sys
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

filename = sys.argv[1]
col_0_name = sys.argv[2]
y_label_0 = sys.argv[3]
x_label_0 = sys.argv[4]
title_0 = sys.argv[5]
y_label_1 = sys.argv[6]
x_label_1 = sys.argv[7]
title_1 = sys.argv[8]
minX = sys.argv[9]
maxX = sys.argv[10]
minY = sys.argv[11]
maxY = sys.argv[12]

df = pd.read_csv(filename, header=None, names=[col_0_name, 'hfInsert', 'hfQuery', 'cuInsert', 'cuQuery', 'thSort', 'thSearch'])


# In[28]:


fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(df[col_0_name], df['hfInsert'], label='HashFight', marker='o')
ax1.plot(df[col_0_name], df['cuInsert'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df[col_0_name], df['thInsert'], label='Thrust Sort', marker='x')
ax1.legend(loc=2)
plt.ylabel(y_label_0)
plt.xlabel(x_label_0)
plt.axis([minX, maxX, minY, maxY])
plt.title(title_0)
#plt.yscale('log')

# In[29]:


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)

ax1.plot(df[col_0_name], df['hfQuery'], label='HashFight', marker='o')
ax1.plot(df[col_0_name], df['cuQuery'], label='CUDPP Cuckoo', linestyle='--')
ax1.plot(df[col_0_name], df['thQuery'], label='Thrust Search', marker='x')
ax1.legend(loc=2)
plt.ylabel(y_label_1)
plt.xlabel(x_label_1)
plt.axis([minX, maxX, minY, maxY])
plt.title(title_1)
#plt.yscale('log')

plt.show()

# In[13]:


df

