#!/usr/bin/env python
# coding: utf-8

# In[1]:


#HW5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import math


# In[2]:


X = np.arange(-1,1,0.02).reshape(-1, 1)
mu = np.zeros(X.shape)


# In[3]:


def g_kernel(X1, X2, sigma=1):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2*(X1@X2.T)
    return np.exp(-1*sqdist/(2*sigma)**2)


# In[4]:


def e_kernel(X1, X2, theta):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2*(X1@X2.T)
    return np.exp(-theta*np.sqrt(sqdist))


# In[5]:


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    #plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, ls='--', label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    #plt.legend()


# In[6]:


cov_g = g_kernel(X, X, 1)
samples = np.random.multivariate_normal(mu.ravel(), cov_g, 5)
plot_gp(mu, cov_g, X, samples=samples)


# In[7]:


cov_e = e_kernel(X, X, 1)
samples = np.random.multivariate_normal(mu.ravel(), cov_e, 5)
plot_gp(mu, cov_e, X, samples=samples)


# In[8]:


X_new = np.arange(0,1,0.01).reshape(-1, 1)


# In[9]:


y = np.sin(X_new*3.14*2)
plt.plot(X_new, y, lw=1, color='green',label='sinx')
plt.show()


# In[10]:


noise = np.random.normal(0, 0.5, size=100)
t = y + noise.reshape(-1,1)


# In[11]:


plt.plot(X_new, y, lw=1, color='green',label='sinx')
plt.scatter(X_new, t, c='b', marker='o', label='noise', s=50)
plt.show()


# In[12]:


C = g_kernel(X_new, X_new, 0.3)


# In[13]:


for i in range(100):
    C[i][i] = C[i][i] + np.var(noise)


# In[14]:


X_test = np.arange(0,1,0.01).reshape(-1, 1)
k = g_kernel(X_new, X_test, 0.3)


# In[15]:


y_predict = k.T@np.linalg.inv(C)@t


# In[16]:


plt.plot(X_new, y, lw=1, color='green',label='sinx')
plt.plot(X_test, y_predict, lw=1, color='red',label='prediction')
X_t = X_test.ravel()
y_p = y_predict.ravel()
c = g_kernel(X_test, X_test, 0.3) - y_predict
uncertainty = np.sqrt(np.diag(c))
plt.fill_between(X_t, y_p + uncertainty, y_p - uncertainty, facecolor='red', alpha=0.1)
plt.scatter(X_new[0], t[0], c='b', marker='o',  s=50)
plt.scatter(X_new[10], t[10], c='b', marker='o', s=50)
plt.scatter(X_new[20], t[20], c='b', marker='o', s=50)
plt.scatter(X_new[30], t[30], c='b', marker='o',  s=50)
plt.scatter(X_new[40], t[40], c='b', marker='o', s=50)
plt.scatter(X_new[50], t[50], c='b', marker='o', s=50)
plt.scatter(X_new[60], t[60], c='b', marker='o', s=50)
plt.legend()
plt.show()


# In[17]:


#load the train data
trainingSet = pd.read_csv("../HW2/trainingSet.csv")
trainingLabel = pd.read_csv("../HW2/trainingLabel.csv")
testSet = pd.read_csv("../HW2/testSet.csv")
testLabel = pd.read_csv("../HW2/testLabel.csv")
x = trainingSet["x"]
y = trainingSet["y"]
c = trainingLabel["c"]
data = {'x':x, 'y':y, 'c':c}
data = pd.DataFrame(data)
X = data[['x', 'y']]
y = data['c']
X = np.array(X, np.float)
y = np.array(y, np.float)


# In[18]:


#load the test file
x_test = testSet["x"]
y_test = testSet["y"]
c_test = testLabel["c"]
test = {'x':x_test, 'y':y_test, 'c':c_test}
test = pd.DataFrame(test)
X_test = test[['x', 'y']]
y_test = test['c']
X_test = np.array(X_test, np.float)
y_test = np.array(y_test, np.float)
test1, test2 = [], []
for i in range(len(y_test)):
    if y_test[i] == 1:
        test1.append(X_test[i])
    else:
        test2.append(X_test[i])
test1 = np.array(test1, np.float)
test2 = np.array(test2, np.float)


# In[19]:


sample1 = []
sample2 = []
for i in range(len(y)):
    if y[i] == 1:
        sample1.append(X[i])
    else:
        sample2.append(X[i])
sample1 = np.array(sample1, np.float)
sample2 = np.array(sample2, np.float)


# In[20]:


fig = plt.figure(figsize=(8,8))
plt.scatter(sample1[:, 0], sample1[:, 1], c='blue', s=10, label='class1')
plt.scatter(sample2[:, 0], sample2[:, 1], c='orange', s=10, label='class2')


# In[21]:


fig = plt.figure(figsize=(8,8))
plt.scatter(test1[:, 0], test1[:, 1], c='blue', s=10, label='class1')
plt.scatter(test2[:, 0], test2[:, 1], c='orange', s=10, label='class2')
plt.show()


# In[22]:


C_n = g_kernel(X, X, 0.013)


# In[23]:


a = np.linalg.inv(C_n)@y


# In[24]:


y_pred = g_kernel(X, X_test, 0.013).T@a


# In[25]:


temp = 1/(1+np.exp(-y_pred))
temp[temp<0.5] = -1
temp[temp>=0.5] = 1


# In[26]:


fig = plt.figure(figsize=(8,8))
mask = temp == 1
plt.scatter(X_test[:, 0][mask], X_test[:, 1][mask], c='blue', s=10, label='class1')
plt.scatter(X_test[:, 0][~mask], X_test[:, 1][~mask], c='orange', s=10, label='class2')
plt.show()
unique, counts = np.unique((temp == y_test), return_counts=True)
acc = dict(zip(unique, counts))[1]/len(temp)
print("accuarcy is ", acc)

