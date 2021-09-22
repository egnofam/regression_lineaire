#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


# In[9]:


x,y = make_regression(n_samples=100,n_features=1,noise=10)


# In[10]:


plt.scatter(x,y)


# In[18]:


#model


# In[11]:


print(x.shape)
print(y.shape)


# In[20]:


#resize y 


# In[12]:


y = y.reshape((y.shape[0],1))


# In[13]:


print(x.shape)
print(y.shape)


# In[15]:


#model definition


# In[16]:


X = np.concatenate((x,np.ones((x.shape))),axis=1)


# In[17]:


def model(X,theta):
    return X.dot(theta)


# In[49]:


#initialise theta


# In[18]:


theta = np.random.randn(2,1)
theta


# In[19]:


plt.scatter(x,y)
plt.plot(x,model(X,theta),c='r')


# In[55]:


#cost function


# In[20]:


def cost_function(X,y,theta):
    m=len(y)
    return 1/(2*m)*np.sum(X.dot(theta)-y)**2
    


# In[21]:


cost_function(X,y,theta)


# In[71]:


#gradient


# In[22]:


def gradient(X,y,theta):
    m=len(y)
    return 1/m*X.T.dot((np.dot(X,theta)-y))


# In[23]:


gradient(X,y,theta)


# In[31]:


def gradient_descent(X,y,theta,lr,nb_iterations):
    cost_history = np.zeros(nb_iterations)
    for i in range(0,nb_iterations):
        cost_history[i]=cost_function(X,y,theta)
        theta=theta-lr*gradient(X,y,theta)
    return theta,cost_history


# In[32]:


theta_final,cost_history=gradient_descent(X,y,theta,lr=0.01,nb_iterations=1000)


# In[33]:


plt.scatter(x,y)
plt.plot(x,model(X,theta_final),c='red')


# In[34]:


#plot courbe d'apprentissage


# In[39]:


plt.plot(range(1000),cost_history)


# In[40]:


#evaluation du modele


# In[44]:


def evaluation(y,pred):
    u=((y-pred)**2).sum()
    v = ((y-y.mean())**2).sum()
    return 1 - u/v


# In[45]:


print(evaluation(y,model(X,theta_final)))


# In[ ]:





# In[ ]:




