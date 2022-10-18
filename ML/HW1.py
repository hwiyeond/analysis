#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install -U scikit-learn scipy matplotlib


# In[4]:


pip install pandas


# In[567]:


from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-talk')


# In[568]:


df = pd.read_csv("./csv_file/x_train.csv") #load training set
df.head()
x = df["1"]
y = df["2"]
z = df["3"]
data = {'x': x, 'y': y, 'z':z}
data = pd.DataFrame(data)
Y = data[['y', 'z']] 
#Y = data[['x', 'y']] 
#Y = data[['x', 'z']] 
T = data[['x', 'y', 'z']]


# In[569]:


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
fig = pyplot.figure(figsize=(10,10))
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()


# In[570]:


#estimate the coefficients, A and b of a linear model y = Ax + b
line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1,1), Y)
#line_fitter.fit(z.values.reshape(-1,1), Y)
#line_fitter.fit(y.values.reshape(-1,1), Y)
A = line_fitter.coef_
b = line_fitter.intercept_
print("A value is: ", A)
print("b value is: ", b)


# In[571]:


A = np.array([-0.05178006, -0.23682096], np.float)
x = np.array(x, np.float)
#x = np.array(z, np.float)
#x = np.array(y, np.float)
Y = np.array(Y, np.float)
T = np.array(T, np.float)


# In[572]:


#precision matrix of prior
mu_total = np.array(np.mean(T), np.float)
temp = []
for i in range(6000):
    t = T[i] - mu_total
    tdot = np.dot(t.reshape(-1,1), t.reshape(1,-1))
    temp.append(tdot)

cov_temp = np.array([[0,0,0],[0,0,0],[0,0,0]], np.float)
for t in temp:
    cov_temp += t
cov_matrix = cov_temp/6000
pre_prior = np.linalg.inv(cov_matrix) 
print("covariance matrix of prior : ", cov_matrix)
print("precision matrix of prior : ", pre_prior)


# In[573]:


mu = np.mean(x)
var = np.var(x)
precision_x = 1/var
print("mu of x : ", mu)
print("variance of x : ", var)
print("precision of x : ", precision_x)


# In[575]:


#precision matrix of likelihood
#var_YY
mu_yz = np.array(np.mean(Y), np.float)
temp = []
for i in range(6000):
    t = Y[i] - mu_yz
    tdot = np.dot(t.reshape(-1,1), t.reshape(1,-1))
    temp.append(tdot)
cov_temp = np.array([[0,0],[0,0]], np.float)
for t in temp:
    cov_temp += t
var_YY = cov_temp/6000

#var_Yx
temp = []
mu_x = np.mean(x)
for i in range(6000):
    t = Y[i] - mu_yz
    t2 = x[i] - mu_x
    tdot = np.dot(t.reshape(-1,1), t2.reshape(1,-1))
    temp.append(tdot)
cov_temp = np.array([[0],[0]], np.float)
for t in temp:
    cov_temp += t
var_Yx = cov_temp/6000


# In[576]:


#var_xY
temp = []
mu_x = np.mean(x)
for i in range(6000):
    t = Y[i] - mu_yz
    t2 = x[i] - mu_x
    tdot = np.dot(t2.reshape(-1,1), t.reshape(1,-1))
    temp.append(tdot)
cov_temp = np.array([[0,0]], np.float)
for t in temp:
    cov_temp += t
var_xY = cov_temp/6000

#var_xx
var_xx = np.var(x)


# In[578]:


first = np.dot(var_Yx, var_xx)
second = np.dot(first, var_xY)
var_like = var_YY - second
L = np.linalg.inv(var_like)
print("precision matrix of likelihood : ", L)


# In[579]:


#covariance matrix of posterior
first = np.dot(A.reshape(-1,2), var_like)
second = np.dot(first, A) + var_xx
pre_post = 1/second
print("precision of posterior: ", pre_post[0])


# In[580]:


df = pd.read_csv("./csv_file/x_test.csv") #load test set
x = df["1"]
y = df["2"]
z = df["3"]
data = {'x': x, 'y': y, 'z':z}
data = pd.DataFrame(data)
Y = data[['y', 'z']]
#Y = data[['x', 'y']]
#Y = data[['x', 'z']]
Y = np.array(Y, np.float)
x = np.array(x, np.float)
#x = np.array(z, np.float)
#x = np.array(y, np.float)


# In[582]:


#estimate the most likely value of x1~xN
e1 = []
first = np.dot(A.reshape(-1,2), var_like)
for i in range(len(Y)):
    third = Y[i]-b
    not_updated = np.dot(first, third) + mu/var
    estimated_x1 = np.dot(pre_post, not_updated)
    e1.append(estimated_x1)
    print("original: ", x[i], " | estimated: ", estimated_x1)


# In[583]:


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
fig = pyplot.figure(figsize=(10,10))
ax = Axes3D(fig)
ax.scatter(x, y, z, marker='^')
ax.scatter(e1, y, z, marker='x')
#ax.scatter(x, y, e1, marker='x')
#ax.scatter(x, e1, z, marker='x')
pyplot.show()

