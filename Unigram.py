# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:18:07 2018

@author: Jie Lu
"""
#load data
import pandas as pd
import numpy as np
test_data = pd.read_csv('reviews_te.csv',skiprows=0)
train_data = pd.read_csv("reviews_tr.csv",skiprows=0, nrows=500000)

#define tf representation
def tf(corpus):
    dict_tf = {}
    for doc in range(len(corpus)):
        dict_tf[doc] = dict.fromkeys(set(train_data['text'][doc].split()))
        for term in dict_tf[doc]:
            dict_tf[doc][term] = train_data.text[doc].count(term)
        #using affine expansion
        dict_tf[doc]['INTERCEPT'] = 1
    return dict_tf    

tf_version = tf(train_data)

#first time shuffle
w={}
order = list(range(len(train_data)))
np.random.shuffle(order)

for doc in order:
    x_value = np.array([tf_version[doc].get(key) for key in sorted(tf_version[doc].keys())])
    weight = np.array([w.get(key,0) for key in sorted(tf_version[doc].keys())])
    for key in sorted(tf_version[doc].keys()):
        w[key] = w.get(key,0)
    y = int(train_data.label[doc])
    if y == 0:
        y = -1
    if y*np.inner(x_value, weight) <= 0:
        for term in tf_version[doc]:
            w[term] = y*tf_version[doc][term]+w.get(term,0)

#second time shuffle, record where change happens            
np.random.shuffle(order)
when = []
for doc2 in order:
    x_value = np.array([tf_version[doc2].get(key) for key in sorted(tf_version[doc2].keys())])
    weight = np.array([w.get(key,0) for key in sorted(tf_version[doc2].keys())])
    y = int(train_data.label[doc2])
    if y == 0:
        y = -1
    if y*np.inner(x_value, weight) <= 0:
        when.append(doc2)  
        
times = len(train_data)-len(when)

#update changes
w2 = w.copy()
for doc3 in when:
    x_value = np.array([tf_version[doc3].get(key) for key in sorted(tf_version[doc3].keys())])
    weight = np.array([w.get(key,0) for key in sorted(tf_version[doc3].keys())])
    y = int(train_data.label[doc3])
    if y == 0:
        y = -1
    for term in tf_version[doc3]:
        w[term] = y*tf_version[doc3][term]+w.get(term,0)

w.update((x,(y+w2[x]*times)/(times+1)) for x, y in w.items())
w_final = w

#most weighted and least weighted 10 words
best_words = sorted(w, key=w.get, reverse=True)[:10]
print(sorted(best_words))
worst_words = sorted(w, key=w.get)[:10]
print(sorted(worst_words))

#check train risk
train_result = []
for doc in range(len(train_data)):
    x_value = np.array([tf_version[doc].get(key) for key in sorted(tf_version[doc].keys())])
    weight = np.array([w.get(key,0) for key in sorted(tf_version[doc].keys())])
    if np.inner(x_value, weight) <= 0:
        train_result.append(0)
    else:
        train_result.append(1)
train_risk = 1-np.sum(train_result==train_data.label)/len(train_data)
print("The train risk is: ", train_risk)

#check test risk
test_result = []
tf_test = tf(test_data)
for doc in range(len(test_data)):
    x_value = np.array([tf_test[doc].get(key) for key in sorted(tf_test[doc].keys())])
    weight = np.array([w.get(key,0) for key in sorted(tf_test[doc].keys())])
    if np.inner(x_value, weight) <= 0:
        test_result.append(0)
    else:
        test_result.append(1)
test_risk = 1-np.sum(test_result==test_data.label)/len(test_data)
print("The test risk is: ", test_risk)       















