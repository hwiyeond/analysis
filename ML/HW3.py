#!/usr/bin/env python
# coding: utf-8

# In[154]:


#HW3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import math
style.use('fivethirtyeight')


# In[155]:


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


# In[156]:


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


# In[157]:


sample1 = []
sample2 = []
for i in range(len(y)):
    if y[i] == 1:
        sample1.append(X[i])
    else:
        sample2.append(X[i])
sample1 = np.array(sample1, np.float)
sample2 = np.array(sample2, np.float)


# In[158]:


#A
#recall a Fisher's discriminant analysis
tot = np.array([0,0], np.float)
tot2 = np.array([0,0], np.float)
for x in sample1:
    tot += x
for x in sample2:
    tot2 += x
mu1 = tot/len(sample1)
mu2 = tot2/len(sample2)


# In[159]:


diff1 = sample1 - mu1
diff2 = sample2 - mu2
Sw = np.dot(diff1.T, diff1) + np.dot(diff2.T, diff2)
inv_Sw = np.linalg.inv(Sw)
print("Inverse of S within matrix is ")
print(inv_Sw)


# In[160]:


#calculate a weight vector
w_ = np.dot(inv_Sw, mu2-mu1)
w_


# In[161]:


n1 = len(sample1)
n2 = len(sample2)
m = (n1*mu1 + n2*mu2)/(n1 + n2)


# In[162]:


w0_f = - np.dot(w_, m)/w_[1]
w_f = -w_[0]/w_[1]


# In[163]:


#draw data points by discriminating them by the assigned class color
x = [0, 5]
x_fisher = [(n * w_f - w0_f) for n in x]
fig = plt.figure(figsize=(8,8))
plt.scatter(sample1[:, 0], sample1[:, 1], c='blue', s=10, label='class1')
plt.scatter(sample2[:, 0], sample2[:, 1], c='orange', s=10, label='class2')
plt.legend(fontsize='x-large')
plt.plot(x,x_fisher, c='red', label='Fisher')
plt.legend(fontsize='x-large')
plt.show()


# In[164]:


#B.a
#(generative way) calculate a common covariance matrix
N = len(sample1) + len(sample2)
common_cov = (len(sample1)/N)*np.cov(sample1[:,0], sample1[:,1]) + (len(sample2)/N)*np.cov(sample2[:,0], sample2[:,1])


# In[165]:


#weight vector and bias
inv_cov = np.linalg.inv(common_cov)
w_gen = np.dot(inv_cov, mu2 - mu1)
w0_gen = (-1/2) * np.dot(np.dot(mu1.T, inv_cov) , mu1.T) +  (1/2) * np.dot(np.dot(mu2.T, inv_cov) , mu2.T) + math.log(len(sample1)/len(sample2))


# In[166]:


#calculate a gradient and an intercept
w_g = -w_gen[0]/w_gen[1]
w0_g = -w0_gen/w_gen[1]


# In[167]:


#compare Fisher and Generative model clalssification
x = [0, 5]
x_fisher = [(n * w_f - w0_f) for n in x]
x_gen = [(n * w_g - w0_g) for n in x]
fig = plt.figure(figsize=(8,8))
plt.scatter(sample1[:, 0], sample1[:, 1], c='blue', s=10, label='class1')
plt.scatter(sample2[:, 0], sample2[:, 1], c='orange', s=10, label='class2')
plt.legend(fontsize='x-large')
plt.plot(x,x_fisher, c='red', label='Fisher')
plt.plot(x,x_gen, c='blue', label='Generative')
plt.legend(fontsize='x-small')
plt.show()


# In[168]:


#b. accuracy
pred_F, pred_G = [0] * len(y_test), [0] * len(y_test) 
# test the acuuracy of F and G
for i in range(len(y_test)):
    if X_test[i][1] > X_test[i][0]*w_f - w0_f:
        pred_F[i] = 1
    else:
        pred_F[i] = -1
for i in range(len(y_test)):
    if X_test[i][1] > X_test[i][0]*w_g - w0_g:
        pred_G[i] = 1
    else:
        pred_G[i] = -1
pred_F, pred_G = np.array(pred_F, np.float), np.array(pred_G, np.float)

F_correct, G_correct = 0, 0
for i in range(len(y_test)):
    if y_test[i] == pred_F[i]:
        F_correct += 1
    if y_test[i] == pred_G[i]:
        G_correct +=1
F_acc, G_acc = F_correct/len(y_test), G_correct/len(y_test)
F_acc, G_acc


# In[169]:


cov_1 = np.cov(sample1[:,0], sample1[:,1])
cov_2 = np.cov(sample2[:,0], sample2[:,1])


# In[170]:


#c. weight vector and bias 
#class 1 covariance
inv_cov1 = np.linalg.inv(cov_1)
w_1 = np.dot(inv_cov1, mu2-mu1)
w0_1 = (-1/2) * np.dot(np.dot(mu1.T, inv_cov1) , mu1.T) +  (1/2) * np.dot(np.dot(mu2.T, inv_cov1) , mu2.T) + math.log(len(sample1)/len(sample2))


# In[171]:


#class 1 covariance
inv_cov2 = np.linalg.inv(cov_2)
w_2 = np.dot(inv_cov2, mu2-mu1)
w0_2 = (-1/2) * np.dot(np.dot(mu1.T, inv_cov2) , mu1.T) +  (1/2) * np.dot(np.dot(mu2.T, inv_cov2) , mu2.T) + math.log(len(sample1)/len(sample2))


# In[172]:


#calculate a gradient and an intercept
w_g1 = -w_1[0]/w_1[1]
w0_g1 = -w0_1/w_1[1]
w_g2 = -w_2[0]/w_2[1]
w0_g2 = -w0_2/w_2[1]


# In[173]:


#compare Fisher and Generative model clalssification
x = [0, 5]
x_fisher = [(n * w_f - w0_f) for n in x]
x_gen = [(n * w_g - w0_g) for n in x]
x_gen1 = [(n * w_g1 - w0_g1) for n in x]
x_gen2 = [(n * w_g2 - w0_g2) for n in x]
fig = plt.figure(figsize=(8,8))
plt.scatter(sample1[:, 0], sample1[:, 1], c='blue', s=10, label='class1')
plt.scatter(sample2[:, 0], sample2[:, 1], c='orange', s=10, label='class2')
plt.legend(fontsize='x-large')
plt.plot(x,x_fisher, c='red', label='Fisher')
#plt.plot(x,x_gen, c='blue', label='Shared cov')
plt.plot(x,x_gen1, c='green', label='Cov1')
plt.plot(x,x_gen2, c='purple', label='Cov2')
plt.legend(fontsize='x-small')
plt.show()


# In[174]:


#d. accuracy
pred_G1, pred_G2 = [0] * len(y_test), [0] * len(y_test) 

# test the acuuracy of G1 and G2
for i in range(len(y_test)):
    if X_test[i][1] > X_test[i][0] * w_g1 - w0_g1:
        pred_G1[i] = 1
    else:
        pred_G1[i] = -1

for i in range(len(y_test)):
    if X_test[i][1] > X_test[i][0] * w_g2 - w0_g2:
        pred_G2[i] = 1
    else:
        pred_G2[i] = -1

pred_G1, pred_G2 = np.array(pred_G1, np.float), np.array(pred_G2, np.float)

G1_correct, G2_correct = 0, 0
for i in range(len(y_test)):
    if y_test[i] == pred_G1[i]:
        G1_correct += 1
    if y_test[i] == pred_G2[i]:
        G2_correct += 1

G1_acc, G2_acc = G1_correct/len(y_test), G2_correct/len(y_test)
G1_acc, G2_acc


# In[251]:


def y_n(w, x):
    a = np.dot(w, x).astype("float_") 
    return 1.0 / (1.0 + np.exp(-a))


# In[263]:


def NR_update(Pi, t):
    #initialization
    w_old, w_new = np.zeros(len(Pi[1])), np.zeros(len(Pi[1]))
    N = len(Pi)
    R = [[0] * N for _ in range(N)]
    i = 0
    while i < 100:
        # predictition (y_n)
        pred = []
        for i in range(N):
            pred.append(y_n(w_old, Pi[i]))
        pred = np.array(pred, np.float)
        
        # a derivative of E(w)
        E = np.dot(Pi.T, pred-t)
        
        # matrix R
        for i in range(N):
            for j in range(N):
                if i == j:
                    R[i][j] = pred[i] * (1 - pred[i])
        R = np.array(R, np.float)
        
        # inverse of Hessian matrix
        H = np.dot(np.dot(Pi.T, R), Pi)
        d = np.linalg.det(H)
        H_inv = np.linalg.inv(H)
        
        
        # Newton-Raphson update
        w_new = w_old - np.dot(H_inv, E)
        
        if np.abs(w_new[0] - w_old[0]) > 0.0000001:
            w_old = w_new
        else:
            break
        i += 1
    return w_new


# In[264]:


T, Pi = [], []
for i in range(len(X)):
    Pi.append([X[i][0], X[i][1], 1])

for i in range(len(y)):
    if y[i] == -1:
        T.append(0)
    else:
        T.append(1)
T, Pi = np.array(T, np.float), np.array(Pi, np.float)


# In[280]:


NR_update(Pi, T)


# In[281]:


w1, w2, w3 = NR_update(Pi, T)[0], NR_update(Pi, T)[1], NR_update(Pi, T)[2]


# In[282]:


l = -(w1/w2)
l0 = w3/w2


# In[283]:


x1 = [0, 5]
x2 = [(n * l - l0) for n in x1]
fig = plt.figure(figsize=(8,8))
plt.scatter(test1[:, 0], test1[:, 1], c='blue', s=10, label='class1')
plt.scatter(test2[:, 0], test2[:, 1], c='orange', s=10, label='class2')
plt.legend(fontsize='x-large')
plt.plot(x1, x2, c='green')
plt.show()


# In[284]:


y_predNR = [0] * len(y_test)
for i in range(len(y_test)):
    if X_test[i][1] > X_test[i][0] * l - l0:
        y_predNR[i] = 1
    else:
        y_predNR[i] = -1
y_predNR = np.array(y_predNR, np.float)
NR_correct = 0
for i in range(len(y_test)):
    if y_test[i] == y_predNR[i]:
        NR_correct += 1 
NR_acc = NR_correct / len(y_test)
NR_acc


# In[380]:


# E. build a rbf function that return the data of increased dimenstion (# of data)
def input2rbf(base, input_data, sigma):
    fvs = []
    for i in range(len(base)):
        base_point = base[[i],...]
        fv = np.exp(-(np.linalg.norm(input_data - base_point, axis=1))/(2*sigma**2))
        fvs.append(fv)

    return np.array(fvs)


# In[300]:


train_data = np.array(pd.read_csv('../HW2/trainingSet.csv'))
train_labels = np.squeeze(np.array(pd.read_csv('../HW2/trainingLabel.csv')))
test_data = np.array(pd.read_csv('../HW2/testSet.csv'))
test_labels = np.squeeze(np.array(pd.read_csv('../HW2/testLabel.csv')))

c1_index = np.where(train_labels==-1)[0]
c2_index = np.where(train_labels==1)[0]
train_labels[c1_index] = 0


# In[397]:


new_train_data = input2rbf(train_data, train_data, 1)


# In[398]:


w = NR_update(new_train_data, train_labels)
w.shape


# In[399]:


# make x- and y- grids for visualization
x_grid = np.linspace(0.5,4.5,100)
XX, YY = np.meshgrid(x_grid, x_grid)
grid_p = np.vstack([XX.flatten(), YY.flatten()]).T

#input train data to rbf function 
tests = input2rbf(train_data, grid_p, 1)


# In[400]:


#map the prediction labels
pred_labels = y_n(w, tests)
pred_c1_index = np.where(pred_labels<0.5)[0]
pred_c2_index = np.where(pred_labels>=0.5)[0]


# In[401]:


#Visualization
plt.figure(figsize=(8,8))
plt.plot(grid_p[pred_c1_index,0], grid_p[pred_c1_index,1], '.', color='C2', alpha=0.3)
plt.plot(grid_p[pred_c2_index,0], grid_p[pred_c2_index,1], '.', color='C0', alpha=0.3)
plt.plot(train_data[c1_index, 0], train_data[c1_index, 1], 'o', color='C2')
plt.plot(train_data[c2_index, 0], train_data[c2_index, 1], 'o', color='C0')
plt.grid(True)
plt.xlim([0.5, 4.5])
plt.ylim([0.5, 4.5])
plt.show()

