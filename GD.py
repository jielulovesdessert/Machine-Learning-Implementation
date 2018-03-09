# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:58:56 2018

@author: Jie Lu
"""

import scipy.io
import numpy as np
data = scipy.io.loadmat('hw3data.mat')

#get some stats from the data, would be used for (c)
avg_1 = np.average(data['data'].T[0])
avg_2 = np.average(data['data'].T[1])
avg_3 = np.average(data['data'].T[2])
std_1 = np.std(data['data'].T[0])
std_2 = np.std(data['data'].T[1])
std_3 = np.std(data['data'].T[2])
print('feature 1 avg:',avg_1,'feature 1 std:',std_1)
print('feature 2 avg:',avg_2,'feature 2 std:',std_2)
print('feature 3 avg:',avg_3,'feature 3 std:',std_3)
print('feature 1 range:',np.min(data['data'].T[0]),np.max(data['data'].T[0]))
print('feature 2 range:',np.min(data['data'].T[1]),np.max(data['data'].T[1]))
print('feature 3 range:',np.min(data['data'].T[2]),np.max(data['data'].T[2]))

beta = np.zeros(3)
beta0 = 0
labels = data['labels']
data_1 = data['data']
target = 0.65064

def value(data, label, beta, beta0):
    value = 0 
    for i in range(len(data)):
        value += np.log(1 + np.exp(beta0 + np.dot(data[i], beta)))-label[i]*(beta0 + np.dot(data[i], beta))
    return value/data.shape[0]

def delta(data, label, beta, beta0):
    delta0 = 0
    delta1 = 0
    for i in range(len(data)):
        expo = np.exp(beta0 + np.dot(data[i], beta))
        delta0 += expo /(1+expo) 
        delta0 -= label[i]
        delta1 += data[i] * (expo /(1+expo))   
        delta1 -= data[i] *label[i]
    delta0 /= data.shape[0]
    delta1 /= data.shape[0]
    return delta0, delta1

def learning(data, label,target):
    beta = np.zeros(3)
    beta0 = 0
    step = 1.0
    time = 0
    obj = value(data, label, beta, beta0)
    while value(data, label, beta, beta0) > target:
        delta0, delta1 = delta(data, label, beta, beta0)
        beta_new = beta - step * delta1
        beta0_new = beta0 - step * delta0[0]
        while value(data, label, beta_new, beta0_new) > obj - 0.5 * step * (sum(delta1 * delta0[0]) + delta1*delta0[0]):
            step /= 2
        beta0 -= step * delta0[0]
        beta -= step * delta1     
        obj =  value(data, label, beta, beta0)       
        time  += 1
    return time

time_1 = learning(data_1, labels, target)
print(time_1)

#scale
for i in range(len(data['data'])):
    data_1[i][0] = data_1[i][0]/std_1
    data_1[i][1] = data_1[i][1]/std_2
    data_1[i][2] = data_1[i][2]/std_3

time_2 = learning(data_1, labels, target)
print(time_2)   

def valid_error(data, label, beta, beta0):
    error = 0
    for i in range(len(data)): 
        if np.dot(data[i], beta) + beta0 <= 0:
            pred = 0
        else:
            pred = 1
        if pred != label[i]:
            error += 1     
    return error / len(data)

def learning2(data, label):
    time = 0
    beta = np.zeros(3)
    beta0 = 0
    step = 1.0
    train_data = data[:data.shape[0]*0.8] 
    train_label = label[:data.shape[0]*0.8]
    valid_data = data[data.shape[0]*0.8:]
    valid_label = label[data.shape[0]*0.8:]
    obj = value(train_data, train_label, beta, beta0)
    error = float("inf")
    while True:
        delta0, delta1 = delta(train_data, train_label, beta, beta0)
        beta_new = beta - step * delta1
        beta0_new = beta0 - step * delta0[0]
        while value(train_data, train_label, beta_new, beta0_new) > obj - 0.5 * step * (sum(delta1 * delta0[0]) + delta1*delta0[0]):
            step = step / 2
        time  += 1
        beta0 -= step * delta0[0]
        beta -= step * delta1
        obj =  value(train_data, train_label, beta, beta0)
        if type(np.log2(time)) is int and np.log2(time)>=5:
            valid_err = valid_error(valid_data, valid_label, beta, beta0)
            if valid_err > 0.99 * error:
                return time, value(train_data, train_label, beta, beta0), valid_err
            error = min(error, valid_err)

time_3, final_value_1, valid_error_1 = learning2(data_1, labels)
print('For the transformed data:',time_3, final_value_1, valid_error_1)

time_4, final_value_2, valid_error_2 = learning2(data['data'], labels)
print('For the origin data:', time_4, final_value_2, valid_error_2)








