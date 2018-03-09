# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:18:08 2018

@author: Jie Lu
"""

# load data
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
test_data = pd.read_csv('reviews_te.csv',skiprows=0)
train_data = pd.read_csv("reviews_tr.csv",skiprows=0, nrows=500000)

# preparation
bag = TfidfVectorizer(analyzer='word')
train = bag.fit_transform(train_data.text)
tfidf = sparse.hstack((train,
                          np.ones(train.shape[0])[:,None])).tocsr()
labels = 2*train_data.label.values-1
index = np.arange(tfidf.shape[0])
w = sparse.csr_matrix((1, tfidf.shape[1]), dtype=np.float64)
avg = None

# 2 times shuffle 
for i in range(2):
    np.random.shuffle(index) 
    for j in range(len(index)):
        predictor = np.sign(tfidf[index[j],:].dot(w.T)[0,0])
        if i == 0 and j == len(index) - 1:
            avg = w
        if predictor*labels[index[j]] != 1: 
            w  += labels[index[j]] * tfidf[index[j],:]
        if avg != None:
            avg += w
w_avg = avg/(len(index) + 1)

# Check the train risk
prediction = tfidf.dot(w_avg.T).sign()
train_risk = 1 - np.mean((prediction.T.toarray()*labels+ 1)/2)
print("train risk: ", train_risk)

# Check the test risk
test = bag.transform(test_data.text)
tfidf_test = sparse.hstack((test,
                            np.ones(test.shape[0])[:,None])).tocsr()
labels_test = test_data.label.values
pred_test = tfidf_test.dot(w_avg.T).sign()
test_risk = 1 - np.mean((pred_test.T.toarray()*labels_test+1)/2)
print("test risk: ", test_risk)













