#!/usr/bin/env python
# coding: utf-8

# In[1]:


#HW6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.font_manager as fm
import random as rd


# In[749]:


m1 = np.array([1, 2])
m2 = np.array([-1, 1])
m3 = np.array([-1, 4])
cov = np.array([[0.3, 0],[0, 0.3]])


# In[750]:


sample1 = np.random.multivariate_normal(m1, cov, 150)
sample2 = np.random.multivariate_normal(m2, cov, 150)
sample3 = np.random.multivariate_normal(m3, cov, 150)


# In[751]:


fig = plt.figure(figsize=(8,6))
plt.scatter(sample1[:,0], sample1[:,1], c='g', marker='o', label='sample 1', s=10)
plt.scatter(sample2[:,0], sample2[:,1], c='b', marker='o', label='sample 2', s=10)
plt.scatter(sample3[:,0], sample3[:,1], c='r', marker='o', label='sample 3', s=10)
plt.legend()
plt.show()


# In[752]:


#c1 = np.random.rand(2)
#c2 = np.random.rand(2)
#c3 = np.random.rand(2)


# In[753]:


data = np.concatenate((sample1, sample2, sample3), axis=0)


# In[754]:


fig = plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], c='black',s=10)
plt.show()


# In[755]:


#number of clusters
K = 3
#initialize centers 
center = np.zeros((K,2))
for i in range(K):
    center[i] = np.random.rand(2)


# In[787]:


fig = plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], c='black',s=10)
plt.scatter(center[0][0], center[0][1], c='b',s=30)
plt.scatter(center[1][0], center[1][1], c='r',s=30)
plt.scatter(center[2][0], center[2][1], c='g',s=30)
plt.show()


# In[788]:


#EM algorithm
temp_c = np.zeros(len(data))
array1, array2, array3 = [], [], []
for _ in range(1):
    #세 mean까지의 거리 계산하는 과정
    for j in range(len(data)):
        tempDist = np.zeros(K)
        for i in range(K):
            # 3점에 대해 거리 계산해서 tempDist의 같은 index에 저장
            tempDist[i] = (center[i] - data[j])@(center[i] - data[j])
        # 3개의 거리 중 최솟값의 index를 temp_c의 같은 index에 저장
        temp_c[j] = np.where(min(tempDist) == tempDist)[0][0]
        if temp_c[j] == 0:
            array1.append(data[j])
        elif temp_c[j] == 1:
            array2.append(data[j])
        else:
            array3.append(data[j])
    #update the means
    c_old = center
    center = np.array([np.mean(array1, axis=0), np.mean(array2, axis=0), np.mean(array3, axis=0)])
    
    #if (c_old[0] - np.mean(array1, axis=0))@(c_old[0] - np.mean(array1, axis=0)) < 0.0001:
    #    break
array1, array2, array3 = np.array(array1), np.array(array2), np.array(array3)


# In[766]:


array3.shape


# In[767]:


fig = plt.figure(figsize=(8,6))
plt.scatter(array1[:,0], array1[:,1], c='b', marker='o',  s=10)
plt.scatter(array2[:,0], array2[:,1], c='g', marker='o',  s=10)
plt.scatter(array3[:,0], array3[:,1], c='r', marker='o',  s=10)
plt.scatter(center[0][0], center[0][1], c='black',marker='o',s=50)
plt.scatter(center[1][0], center[1][1], c='black',marker='o',s=50)
plt.scatter(center[2][0], center[2][1], c='black',marker='o',s=50)
plt.show()


# In[943]:


#Online algorithm
temp_c = np.zeros(len(data))
array1, array2, array3 = [], [], []
iter=0
m = 0.5
count = 1
#initialize centers 
center = np.zeros((K,2))
for i in range(K):
    center[i] = np.random.rand(2)


# In[956]:


#EM algorithm
temp_c = np.zeros(len(data))
array1, array2, array3 = [], [], []
for _ in range(1):
    #세 mean까지의 거리 계산하는 과정
    for j in range(len(data)):
        tempDist = np.zeros(K)
        for i in range(K):
            # 3점에 대해 거리 계산해서 tempDist의 같은 index에 저장
            tempDist[i] = (center[i] - data[j])@(center[i] - data[j])
        # 3개의 거리 중 최솟값의 index를 temp_c의 같은 index에 저장
        temp_c[j] = np.where(min(tempDist) == tempDist)[0][0]
        if temp_c[j] == 0:
            array1.append(data[j])
        elif temp_c[j] == 1:
            array2.append(data[j])
        else:
            array3.append(data[j])
    #update the means
    c_old = center
    center = np.array([np.mean(array1, axis=0), np.mean(array2, axis=0), np.mean(array3, axis=0)])
    
array1, array2, array3 = np.array(array1), np.array(array2), np.array(array3)


# In[653]:


#2 Mixture of Gaussians
fig = plt.figure(figsize=(8,6))
plt.scatter(sample1[:,0], sample1[:,1], c='b', marker='o', label='sample 1', s=10)
plt.scatter(sample2[:,0], sample2[:,1], c='r', marker='o', label='sample 2', s=10)
plt.scatter(sample3[:,0], sample3[:,1], c='g', marker='o', label='sample 3', s=10)
plt.legend()
plt.show()


# In[950]:


#number of clusters
K = 3
N = len(data)
#initialize centers
center = np.zeros((K,2))
for i in range(K):
    center[i] = np.random.rand(2)
print(center)


# In[960]:


# input given data X
# output pi, mu, cov (1~K)
# ramdonly initialize pi
array1, array2, array3 = [], [], []
for j in range(len(data)):
    tempDist = np.zeros(K)
    for i in range(K):
        # 3점에 대해 거리 계산해서 tempDist의 같은 index에 저장
        tempDist[i] = (center[i] - data[j])@(center[i] - data[j])
    # 3개의 거리 중 최솟값의 index를 temp_c의 같은 index에 저장
    temp_c[j] = np.where(min(tempDist) == tempDist)[0][0]
    if temp_c[j] == 0:
        array1.append(data[j])
    elif temp_c[j] == 1:
        array2.append(data[j])
    else:
        array3.append(data[j])
array1, array2, array3 = np.array(array1), np.array(array2), np.array(array3)


# In[961]:


array1.shape, array2.shape, array3.shape


# In[962]:


def normal_dist(x, mu, var):
    return ((2*np.pi)**(-len(x)/2)*np.linalg.det(var))**(-1/2) * np.exp(-((x-mu)@np.linalg.inv(var)@(x-mu).T)/2)
    #return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)


# In[954]:


#mu, cov, pi initialize
c1 = np.atleast_2d(np.cov(array1.T))
c2 = np.atleast_2d(np.cov(array2.T))
c3 = np.atleast_2d(np.cov(array3.T))
n1 = np.nan_to_num(normal_dist(data, center[0], c1), copy=False)
n2 = np.nan_to_num(normal_dist(data, center[1], c2), copy=False)
n3 = np.nan_to_num(normal_dist(data, center[2], c3), copy=False)
pi = np.array([len(array1)/len(data), len(array2)/len(data), len(array3)/len(data)])
ln_P = np.sum(np.log(pi[0]*n1 + pi[1]*n2 + pi[2]*n3))
print(ln_P)


# In[959]:


#E-step: evaluate the responsibilities
normal_1 = np.nan_to_num(normal_dist(data, center[0], c1), copy=False)
normal_2 = np.nan_to_num(normal_dist(data, center[1], c2), copy=False)
normal_3 = np.nan_to_num(normal_dist(data, center[2], c3), copy=False)

tot_r = pi[0]*normal_1+ pi[1]*normal_2+ pi[2]*normal_3
r1 = pi[0] * normal_1 / (tot_r+ 1e-7)
r2 = pi[1] * normal_2 / (tot_r+ 1e-7)
r3 = pi[2] * normal_3 / (tot_r+ 1e-7)

#M-step: update the parameters
N1, N2, N3 = np.sum(r1)+ 1e-7, np.sum(r2)+ 1e-7, np.sum(r3)+ 1e-7

#----update the mean
center = np.array([np.sum(r1@data, axis=0)/N1, np.sum(r2@data, axis=0)/N2, np.sum(r3@data, axis=0)/N3])
print(center)

#----update the covariance
c1 = ((r1@(data-center[0])).T)@(data-center[0])/N1
c2 = ((r2@(data-center[1])).T)@(data-center[1])/N2
c3 = ((r3@(data-center[2])).T)@(data-center[2])/N3

#----update the pi
N = N1 + N2 + N3
pi = np.array([N1/N, N2/N, N3/N])

#----Evaluate the log-likelihood
ln_P = np.sum(np.log(pi[0]*normal_1 + pi[1]*normal_2 + pi[2]*normal_3))
print(ln_P)


# In[957]:


fig = plt.figure(figsize=(8,6))
plt.scatter(array1[:,0], array1[:,1], c='b', marker='o',  s=10)
plt.scatter(array2[:,0], array2[:,1], c='g', marker='o',  s=10)
plt.scatter(array3[:,0], array3[:,1], c='r', marker='o',  s=10)
plt.show()


# In[ ]:




