#!/usr/bin/env python
# coding: utf-8

# In[286]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[350]:


#HW2-1A
mu_1 = np.array([1,3], np.float)
mu_2 = np.array([3,2], np.float)
s_1 = np.array([[2,-1.5], [-1.5, 2]], np.float)
s_2 = np.array([[1, 0], [0, 1]], np.float)


# In[351]:


#Random sampling of class 1 and 2 from each mu and variance
sample_1 = np.random.multivariate_normal(mu_1, s_1, 100)
sample_2 = np.random.multivariate_normal(mu_2, s_2, 100)


# In[364]:


#draw data points by discriminating them by the assigned class color
fig = plt.figure(figsize=(8,8))
plt.scatter(sample_1[:, 0], sample_1[:, 1], c='blue', s=10, label='class1')
plt.scatter(sample_2[:, 0], sample_2[:, 1], c='orange', s=10, label='class2')
plt.legend(fontsize='x-large')
plt.show()


# In[365]:


#S_within
s_within = s_1 + s_2
inv_s_within = np.linalg.inv(s_within)
print("inverse of S within matrix: ")
print(inv_s_within)


# In[366]:


#S_between and weight vector
b = mu_2 - mu_1
w = np.dot(inv_s_within, b.reshape(2,-1))
w = w[0][0]
d = -1/w


# In[383]:


#Draw decision boundary line and weight vector 
x = [-4, 0, 5]
x2 = [-2, 0, 5]
y = [(d*n) for n in x]
y2 = [(w*n) for n in x2]
fig = plt.figure(figsize=(8,8))
plt.scatter(sample_1[:, 0], sample_1[:, 1], c='blue', s=10, label='class1')
plt.scatter(sample_2[:, 0], sample_2[:, 1], c='orange', s=10, label='class2')
plt.plot(x,y, label='weight vector')
plt.plot(x2,y2, c='green', linestyle='--', label = 'decision boundary line')
plt.legend(fontsize='x-large')
plt.show()

#Check accuracy
false_c1 = 0
false_c2 = 0
for i in range(100):
    #C1인데 C2로 분류된 경우
    if sample_1[i][1] < w*sample_1[i][0]:
        false_c1 += 1
    #C2인데 C1로 분류된 경우
    if sample_2[i][1] > w*sample_2[i][0]:
        false_c2 += 1
print("number of c1 that are classified as c2: ", false_c1)
print("number of c2 that are classified as c1: ", false_c2)
print("total accuracy", 1-(false_c1+false_c2)/100)


# In[407]:


#HW2-1B
import math
q = 1.5
r = np.array([[math.cos(q), -math.sin(q)], [math.sin(q), math.cos(q)]], np.float)
mu_r = np.dot(r,mu_1)

#let's make rotated samples of class 1 data set
sample_r = []
for i in range(len(sample_1)):
    sample_r.append(np.dot(r, sample_1[i]))
sample_r = np.array(sample_r, np.float)

#calculate a S_between and weight vector with the rotated mu_1 (mu_r)
b_r = mu_2 - mu_r
w_r = np.dot(inv_s_within, b_r.reshape(2,-1))
w_r = w_r[0][0]
d_r = -1/w_r

x = [-4, 0, 5]
x2 = [-2, 0, 5]
y = [(d_r*n) for n in x]
y2 = [(w_r*n) for n in x2]
fig = plt.figure(figsize=(8,8))
plt.scatter(sample_1[:, 0], sample_1[:, 1], alpha=0.3, c='blue', s=10, label='class1')
plt.scatter(sample_r[:, 0], sample_r[:, 1], c='red', s=10, label='class 1 rotated')
plt.scatter(sample_2[:, 0], sample_2[:, 1], c='orange', s=10, label='class2')
plt.plot(x,y, label='weight vector')
plt.plot(x2,y2, c= plt.cm.rainbow(0.9),linestyle='--', label = 'decision boundary line')
plt.legend(fontsize='x-large')
plt.show()
false_c1_r = 0
false_c2 = 0
for i in range(100):
    #C1인데 C2로 분류된 경우
    if sample_r[i][1] < w_r*sample_r[i][0]:
        false_c1_r += 1
    #C2인데 C1로 분류된 경우
    if sample_2[i][1] > w_r*sample_2[i][0]:
        false_c2 += 1
print("number of c1 that are classified as c2: ", false_c1_r)
print("number of c2 that are classified as c1: ", false_c2)
print("total accuracy", 1-(false_c1_r+false_c2)/100)
print("degree: ", q*180/math.pi)


# In[408]:


#HW2-2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[409]:


#load the train data
trainingSet = pd.read_csv("trainingSet.csv")
trainingLabel = pd.read_csv("trainingLabel.csv")
testSet = pd.read_csv("testSet.csv")
testLabel = pd.read_csv("testLabel.csv")
x = trainingSet["x"]
y = trainingSet["y"]
c = trainingLabel["c"]
data = {'x':x, 'y':y, 'c':c}
data = pd.DataFrame(data)
X = data[['x', 'y']]
y = data['c']
X = np.array(X, np.float)
y = np.array(y, np.float)


# In[410]:


#train the data
clf = LinearDiscriminantAnalysis()
clf.fit(X,y)


# In[411]:


#load the teat data
x_test = testSet["x"]
y_test = testSet["y"]
c_test = testLabel["c"]
test = {'x':x_test, 'y':y_test, 'c':c_test}
test = pd.DataFrame(test)
X_test = test[['x', 'y']]
y_test = test['c']
X_test = np.array(X_test, np.float)
y_test = np.array(y_test, np.float)


# In[412]:


#test the result with the testSet
correct = 0
y_pred = clf.predict(X_test)
#Calculate the accuracy (1)
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        correct += 1
accuracy = correct/len(y_pred)
print("accuracy is ", accuracy)


# In[413]:


#Calculate the accuracy (2)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy is ' + str(accuracy_score(y_test, y_pred)))

