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
d2v_small_model_name = './model/word_embedding/Doc2vec_new_small2.model'

doc_vectorizer = Doc2Vec.load(d2v_model_name)
doc_vectorizer_small = Doc2Vec.load(d2v_small_model_name)
# doc_vectorizer_small.wv.vectors.shape


data = pd.read_excel('./data/doc_set_final_version3.xlsx')
data['token'] = data.token.apply(lambda x: literal_eval(x))

big_class = np.unique(data.new_class)
small_class = np.unique(data.new_small_class)
big_class_dict  = {i: k for i, k in enumerate(big_class)}
small_class_dict  = {i: k for i, k in enumerate(small_class)}

class_dict = {
    key : list(np.unique(data.loc[data.new_class == key, 'new_small_class']))
    for key in big_class
}


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

TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument(d, c) for d, c in zip(train_X2, train_y)]
tagged_test_docs = [TaggedDocument(d, c) for d, c in zip(test_X2, test_y)]
X_train = [doc_vectorizer_small.infer_vector(doc.words) for doc in tagged_train_docs]
X_test  = [doc_vectorizer_small.infer_vector(doc.words) for doc in tagged_test_docs]


### LogisticRegression
filename_big = './model/ml_model/LR_clf_last.sav'
LR_clf =  pickle.load(open(filename_big, 'rb'))
y_pred = LR_clf.predict(X_test)
y_pred_prob = LR_clf.predict_proba(X_test)
big_pred_prob = np.argsort(-y_pred_prob, axis=1)

print("Logistic Regression 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y)))
print("Logistic Regression 1class Recall: {:.4f}".format(recall_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class precision_score: {:.4f}".format(precision_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class f1_score: {:.4f}".format(f1_score(test_y, y_pred, average='weighted')))
two_acc_small = return_two_calss_acc(LR_clf, X_test, test_y)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))
two_pred = pd.DataFrame(big_pred_prob[:,0:2], columns=['predicted_label1', 'predicted_label2'], index=test_X.index)

filename_small = './model/ml_model/LR_clf_small_last.sav'
LR_clf_small = pickle.load(open(filename_small, 'rb'))
y_pred_small = LR_clf_small.predict(X_test)
y_pred_prob_small = LR_clf_small.predict_proba(X_test)
small_predict_prob = np.argsort(-y_pred_prob_small, axis=1)

print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred_small, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred_small, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred_small, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred_small, average='weighted')))

two_acc_small = return_two_calss_acc(LR_clf_small, X_test, test_y_small)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))

test_X['new_class'] = test_y
test_X = pd.concat([test_X, two_pred], axis=1)

test_X['predicted_label1'] = test_X.predicted_label1.map(big_class_dict)
test_X['predicted_label2'] = test_X.predicted_label2.map(big_class_dict)

small_class_dic = {LR_clf_small.classes_[i]: i for i in range(len(LR_clf_small.classes_))}
key_list = list(small_class_dic.keys())
val_list = list(small_class_dic.values())

score = []
predicted_small_label1 = []
predicted_small_label2 = []
for i in range(len(test_X)):
    sc_candidate = class_dict[test_X.iloc[i]['predicted_label1']] + class_dict[test_X.iloc[i]['predicted_label2']]
    sc_candidate = [x for x in sc_candidate if x in key_list]
    sc_candidate_idx = list({key: small_class_dic[key] for key in sc_candidate}.values())
    sc_candidate2 = [x for x in small_predict_prob[i] if x in sc_candidate_idx]
    first = sc_candidate2[0]
    second = sc_candidate2[1]
    label = list([key_list[val_list.index(first)], key_list[val_list.index(second)]])
    predicted_small_label1.append(key_list[val_list.index(first)])
    predicted_small_label2.append(key_list[val_list.index(second)])
    if test_y_small[i] in label :
        score.append(1)
    else :
        score.append(0)

acc = sum(score)/len(score)
print("hierachy accuracy_score", acc)

print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(predicted_small_label1, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, predicted_small_label1, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, predicted_small_label1, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, predicted_small_label1, average='weighted')))
