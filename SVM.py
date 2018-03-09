# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:37:02 2018

@author: Jie Lu
"""
#load data
import scipy.io
import numpy as np
data = scipy.io.loadmat('hw3data.mat')

#set labels
data['labels'] = 2*data['labels']-1
data['labels'] = data['labels'].astype(np.int8, copy = True)

a = np.asarray([0.0] * len(data['data']))
for k in range(2):
    for i in range(len(a)):
        alpha = 1
        for j in range(len(a)):
            if  i != j:       
                alpha -= 2.0*data['labels'][j]* data['labels'][i]*a[j]*np.dot(data['data'][j], data['data'][i])
        alpha /= 2*np.dot(data['data'][i], data['data'][i])    
        alpha = min(max(0, alpha), 10/len(data['data']))   
        a[i] = alpha   
        
value = np.sum(a)
for i in range(len(a)):
    if a[i] != 0:
        temp = 0
        step = data['labels'][i]*a[i]
    for j in range(len(a)):
        if a[j] != 0:
            temp += data['labels'][j] * a[j] * np.dot(data['data'][i], data['data'][j])
    value -= temp * step

w = 0
for i in range(len(a)):
    if a[i] != 0:
        w += a[i] * data['labels'][i] * data['data'][i]

print (value, w)
