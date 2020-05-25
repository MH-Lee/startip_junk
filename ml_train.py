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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import namedtuple
from sklearn.model_selection import cross_val_score
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
d2v_model_small_name = './model/word_embedding/Doc2vec_new_small.model'
d2v_model_small_name2 = './model/word_embedding/Doc2vec_new_small2.model'

doc_vectorizer = Doc2Vec.load(d2v_model_name)
doc_vectorizer.wv.vectors.shape

doc_vectorizer_small = Doc2Vec.load(d2v_model_small_name2)
doc_vectorizer_small.wv.vectors.shape


data = pd.read_excel('./data/doc_set_final_version3.xlsx')
data['token'] = data.token.apply(lambda x: literal_eval(x))
X_data = data[['token', 'new_small_class']]
target_big = data.new_class.tolist()
target_small = data.new_small_class.tolist()

train_X, test_X, train_Y, test_Y = train_test_split(X_data, target_big,
                                                    test_size=0.3,
                                                    stratify=target_big,
                                                    shuffle=True,
                                                    random_state=1234)

train_y_small = train_X.new_small_class.tolist()
train_X = train_X.token.tolist()
test_y_small = test_X.new_small_class.tolist()
test_X = test_X.token.tolist()

TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument(d, c) for d, c in zip(train_X, train_Y)]
tagged_test_docs = [TaggedDocument(d, c) for d, c in zip(test_X, test_Y)]
X_train = [doc_vectorizer_small.infer_vector(doc.words) for doc in tagged_train_docs]
X_test  = [doc_vectorizer_small.infer_vector(doc.words) for doc in tagged_test_docs]


### LogisticRegression
clf = LogisticRegression(solver='sag',  multi_class='multinomial', random_state=1234)

clf.fit(X_train, train_Y)
filename = './model/ml_model/LR_clf_last.sav'
pickle.dump(clf, open(filename, 'wb'))

LR_scores = cross_val_score(clf, X_train, train_Y, cv=10)
print("Logistics Regression CV Accuracy", LR_scores)
print("Logistics Regression CV 평균 Accuracy: ", np.mean(LR_scores))
print("Logistics Regression CV Std: ", np.std(LR_scores))

y_pred = clf.predict(X_test)
print("Logistic Regression 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_Y)))
print("Logistic Regression 1class Recall: {:.4f}".format(recall_score(test_Y, y_pred, average='weighted')))
print("Logistic Regression 1class precision_score: {:.4f}".format(precision_score(test_Y, y_pred, average='weighted')))
print("Logistic Regression 1class f1_score: {:.4f}".format(f1_score(test_Y, y_pred, average='weighted')))

two_acc = return_two_calss_acc(clf, X_test, test_Y)
print("Logistics Regression 2class 정확도: {:.3f}".format(two_acc))
print()
print()

len(np.unique(train_y_small))

clf.fit(X_train, train_y_small)
filename = './model/ml_model/LR_clf_small_last.sav'
pickle.dump(clf, open(filename, 'wb'))

y_pred = clf.predict(X_test)
print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred, average='weighted')))

two_acc_small = return_two_calss_acc(clf, X_test, test_y_small)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))
