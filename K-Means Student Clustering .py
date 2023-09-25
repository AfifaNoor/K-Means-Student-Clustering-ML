#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


# In[11]:


df=pd.read_csv("student_clustering.csv")


# In[12]:


df


# In[13]:


print('The shape of data is',df.shape)
df.head()


# In[14]:


plt.scatter(df['cgpa'],df['iq'])


# In[15]:


#we have 4 type of student in the data ...


# In[16]:


from sklearn.cluster import KMeans


# In[18]:


wcss=[]

for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)
    


# In[19]:


wcss


# In[ ]:


#wcss decreases with the clustering division number 


# In[ ]:


#wcss-->It is defined as the sum of square distances between the centroids and each points


# In[20]:


plt.plot(range(1,11),wcss)


# In[ ]:


#4 is the Elbow  point where there is no fall fear which means 4 cluster is best for this data .


# In[22]:


x=df.iloc[:,:].values
km=KMeans(n_clusters=4)
y_means=km.fit_predict(x)


# In[23]:


y_means


# In[24]:


#shows rows number with cluster present 3
x[y_means==3,1]


# In[27]:


plt.scatter(x[y_means == 0,0],x[y_means == 0,1],color='blue')
plt.scatter(x[y_means == 1,0],x[y_means == 1,1],color='red')
plt.scatter(x[y_means == 2,0],x[y_means == 2,1],color='green')
plt.scatter(x[y_means == 3,0],x[y_means == 3,1],color='yellow')


# # K-Means on 3-D Data

# In[29]:


from sklearn.datasets import make_blobs

centroids = [(-5,-5,5),(5,5,-5),(3.5,-2.5,4),(-2.5,2.5,-4)]
cluster_std = [1,1,1,1]

X,y = make_blobs(n_samples=200,cluster_std=cluster_std,centers=centroids,n_features=3,random_state=1)


# In[30]:


X


# In[34]:


import plotly.express as px
fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=X[:,2])
fig.show()


# In[35]:


wcss = []
for i in range(1,21):
    km = KMeans(n_clusters=i)
    km.fit_predict(X)
    wcss.append(km.inertia_)


# In[36]:



plt.plot(range(1,21),wcss)


# In[37]:


km = KMeans(n_clusters=4)
y_pred = km.fit_predict(X)


# In[38]:


df = pd.DataFrame()

df['col1'] = X[:,0]
df['col2'] = X[:,1]
df['col3'] = X[:,2]
df['label'] = y_pred


# In[39]:


fig = px.scatter_3d(df,x='col1', y='col2', z='col3',color='label')
fig.show()


# In[ ]:




