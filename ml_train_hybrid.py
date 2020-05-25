###############################################################################
##### 1. Base Package   #######################################################
###############################################################################
import json
import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from time import time
from ast import literal_eval
###############################################################################
##### 2. NLP Package   #######################################################
###############################################################################
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
###############################################################################
##### 3. ML Package   #######################################################
###############################################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import namedtuple
import multiprocessing
import logging
import warnings

def return_two_calss_acc(clf, x_test, y_test):
    y_pred_prob = clf.predict_proba(x_test)
    L = np.argsort(-y_pred_prob, axis=1)
    two_pred = L[:,0:2]

    class_dic = {clf.classes_[i]: i for i in range(len(clf.classes_))}
    key_list = list(class_dic.keys())
    val_list = list(class_dic.values())

    dd = []
    score = []
    for i in range(len(y_test)):
        first = two_pred[i][0]
        second = two_pred[i][1]
        label = list([key_list[val_list.index(first)], key_list[val_list.index(second)]])
        if y_test[i] in label :
            score.append(1)
        else :
            score.append(0)
        dd.append({'y_test':y_test[i], 'first':label[0], 'second': label[1]})
    acc = sum(score)/len(y_test)
    return acc


warnings.filterwarnings("ignore")

d2v_model_name = './model/word_embedding/Doc2vec_new.model'
doc_vectorizer = Doc2Vec.load(d2v_model_name)
doc_vectorizer.wv.vectors.shape

w2v_model_name = './model/word_embedding/Word2vec1(base_token).model'
word_vectorizer = Word2Vec.load(w2v_model_name)
word2vec_matrix = word_vectorizer.wv.vectors
word_index = word_vectorizer.wv.index2word
word2vec_matrix.shape

data = pd.read_excel('./data/doc_set_final_version3.xlsx')
data['token'] = data.token.apply(lambda x: literal_eval(x))
X_data = data[['token', 'new_small_class']]
target_big = data.new_class.tolist()
target_small = data.new_small_class.tolist()

train_X, test_X, train_y, test_y = train_test_split(X_data, target_big,
                                                    test_size=0.3,
                                                    stratify=target_big,
                                                    shuffle=True,
                                                    random_state=1234)

train_y_small = train_X.new_small_class.tolist()
train_X2 = train_X.token.tolist()
test_y_small = test_X.new_small_class.tolist()
test_X2 = test_X.token.tolist()

X_train_d2v = [doc_vectorizer.infer_vector(doc) for doc in train_X2]
X_test_d2v  = [doc_vectorizer.infer_vector(doc) for doc in test_X2]
X_train_d2v = np.array(X_train_d2v)
X_test_d2v = np.array(X_test_d2v)
X_train_d2v.shape
X_test_d2v.shape

tdm = CountVectorizer(analyzer=lambda x: x, vocabulary=word_index)
train_dwm = tdm.fit_transform(train_X2)
X_train_w2v = np.matmul(train_dwm.toarray(), word2vec_matrix)
X_train_w2v.shape

test_dwm = tdm.fit_transform(test_X2)
X_test_w2v = np.matmul(test_dwm.toarray(), word2vec_matrix)
X_test_w2v.shape

X_train_new = X_train_d2v + X_train_w2v
X_test_new = X_test_d2v + X_test_w2v

### LogisticRegression
clf = LogisticRegression(solver='sag',  multi_class='multinomial', random_state=1234)

clf.fit(X_train_new, train_y)
filename = './model/ml_model/LR_clf_hybrid.sav'
pickle.dump(clf, open(filename, 'wb'))

LR_scores = cross_val_score(clf, X_train_new, train_y, cv=10)
print("Logistics Regression CV Accuracy", LR_scores)
print("Logistics Regression CV 평균 Accuracy: ", np.mean(LR_scores))
print("Logistics Regression CV Std: ", np.std(LR_scores))

y_pred = clf.predict(X_test_new)
print("Logistic Regression 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y)))
print("Logistic Regression 1class Recall: {:.4f}".format(recall_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class precision_score: {:.4f}".format(precision_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class f1_score: {:.4f}".format(f1_score(test_y, y_pred, average='weighted')))

two_acc = return_two_calss_acc(clf, X_test_new, test_y)
print("Logistics Regression 2class 정확도: {:.3f}".format(two_acc))
print()
print()


clf.fit(X_train_new, train_y_small)
filename = './model/ml_model/LR_clf_small_hybrid.sav'
pickle.dump(clf, open(filename, 'wb'))

y_pred = clf.predict(X_test_new)
print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred, average='weighted')))

two_acc_small = return_two_calss_acc(clf, X_test_new, test_y_small)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))
