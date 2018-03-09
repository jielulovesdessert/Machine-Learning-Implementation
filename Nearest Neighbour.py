# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 01:48:20 2018

@author: Jie Lu
"""

from scipy.io import loadmat
import random
import numpy as np
import matplotlib.pyplot as plt

ocr = loadmat('ocr.mat')
data = ocr['data']
labels = ocr['labels']
testdata = ocr['testdata']
testlabels = ocr['testlabels']

def NNC (X, y, test):
    x_times_x = np.sum(X*X,axis=1)
    xxmatrix = np.array([x_times_x]*test.shape[0])
    txmatrix = np.matmul(test.astype('float'), X.astype('float').T)
    Euclidean = xxmatrix - 2*txmatrix
    arg = np.argmin(Euclidean, axis=1)
    return y[arg]

n = [1000, 2000, 4000, 8000]
for i in n:
    sel = random.sample(range(60000),i)
    data1 = data[sel].astype('float')
    labels1 = labels[sel]
    result1 = NNC(data1,labels1,testdata)
    print("Test error rate for",i,"training sample "
          "is {:.2f}".format((sum(result1!=testlabels)/len(result1))[0]))

test_error_rate = []
test_error_std = []
for i in n:
    error_rate = []
    for j in range(10):
        sel = random.sample(range(60000),i)
        data1 = data[sel].astype('float')
        labels1 = labels[sel]
        result1 = NNC(data1,labels1,testdata)
        error_rate.append((sum(result1!=testlabels)/len(result1))[0])
    test_error_rate.append(np.mean(error_rate))
    test_error_std.append(np.std(error_rate))

plt.figure()
plt.title("Learning curve for test set")
plt.xlabel("Sample size")
plt.ylabel("Average test error rates")
plt.grid()
plt.fill_between(n,np.array(test_error_rate) - np.array(test_error_std),
                 np.array(test_error_rate) + np.array(test_error_std), alpha=0.1,
                color="r", label="error bar")
plt.plot(n,test_error_rate,'o-', color="r",
         label="Average test error rate")

training_error_rate = []
training_error_std = []
for i in n:
    error_rate = []
    for j in range(10):
        sel = random.sample(range(60000),i)
        data1 = data[sel].astype('float')
        labels1 = labels[sel]
        result1 = NNC(data1,labels1,data1)
        error_rate.append((sum(result1!=labels1)/len(result1))[0])
    training_error_rate.append(np.mean(error_rate))
    training_error_std.append(np.std(error_rate))
    
plt.figure()
plt.title("Learning curve for train set")
plt.xlabel("Sample size")
plt.ylabel("Average training error rates")
plt.grid()
plt.fill_between(n,np.array(training_error_rate) - np.array(training_error_std),
                 np.array(training_error_rate) + np.array(training_error_std), alpha=0.1,
                color="g", label="error bar")
plt.plot(n,training_error_rate,'o-', color="g",
         label="Average test error rate")
plt.show()