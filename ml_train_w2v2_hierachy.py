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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
###############################################################################
##### 3. ML Package   #######################################################
###############################################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import namedtuple, defaultdict
from sklearn.model_selection import cross_val_score
import multiprocessing
import logging
import warnings
warnings.filterwarnings("ignore")

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

data = pd.read_excel('./data/doc_set_final_version3.xlsx')
data['token'] = data.token.apply(lambda x: literal_eval(x))
X_data = data[['token', 'new_small_class']]
target_big = data.new_class.tolist()
target_small = data.new_small_class.tolist()

big_class = np.unique(data.new_class)
small_class = np.unique(data.new_small_class)
big_class_dict  = {i: k for i, k in enumerate(big_class)}
small_class_dict  = {i: k for i, k in enumerate(small_class)}

class_dict = {
    key : list(np.unique(data.loc[data.new_class == key, 'new_small_class']))
    for key in big_class
}



train_X, test_X, train_y, test_y = train_test_split(X_data, target_big,
                                                    test_size=0.3,
                                                    stratify=target_big,
                                                    shuffle=True,
                                                    random_state=1234)

train_y_small = train_X.new_small_class.tolist()
train_X2 = train_X.token.tolist()
test_y_small = test_X.new_small_class.tolist()
test_X2 = test_X.token.tolist()

w2v_model_name = './model/word_embedding/Word2vec1(base_token).model'
word_vectorizer = Word2Vec.load(w2v_model_name)
word2vec_matrix = word_vectorizer.wv.vectors
word_index = word_vectorizer.wv.index2word
word2vec_matrix.shape

tdm = CountVectorizer(analyzer=lambda x: x, vocabulary=word_index)

train_dwm = tdm.fit_transform(train_X2)
train_X_new = np.matmul(train_dwm.toarray(), word2vec_matrix)
train_X_new.shape

test_dwm = tdm.fit_transform(test_X2)
test_X_new = np.matmul(test_dwm.toarray(), word2vec_matrix)
test_X_new.shape


filename = './model/ml_model/LR_clf_w2v.sav'
LR_clf = pickle.load(open(filename, 'rb'))
y_pred = LR_clf.predict(test_X_new)
y_pred_prob = LR_clf.predict_proba(test_X_new)
big_pred_prob = np.argsort(-y_pred_prob, axis=1)
two_pred = pd.DataFrame(big_pred_prob[:,0:2], columns=['predicted_label1', 'predicted_label2'], index=test_X.index)

print("Logistic Regression 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y)))
print("Logistic Regression 1class Recall: {:.4f}".format(recall_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class precision_score: {:.4f}".format(precision_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class f1_score: {:.4f}".format(f1_score(test_y, y_pred, average='weighted')))
two_acc_small = return_two_calss_acc(LR_clf, test_X_new, test_y)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))

filename = './model/ml_model/LR_clf_small_w2v.sav'
LR_clf_small = pickle.load(open(filename, 'rb'))
y_pred_small = LR_clf_small.predict(test_X_new)
y_pred_prob_small = LR_clf_small.predict_proba(test_X_new)
small_predict_prob = np.argsort(-y_pred_prob_small, axis=1)

print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred_small, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred_small, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred_small, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred_small, average='weighted')))

two_acc_small = return_two_calss_acc(LR_clf_small, test_X_new, test_y_small)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))

test_X['new_class'] = test_y
test_X = pd.concat([test_X, two_pred], axis=1)
test_X['predicted_label1'] = test_X.predicted_label1.map(big_class_dict)
test_X['predicted_label2'] = test_X.predicted_label2.map(big_class_dict)

small_class_dic = {LR_clf_small.classes_[i]: i for i in range(len(LR_clf_small.classes_))}
key_list = list(small_class_dic.keys())
val_list = list(small_class_dic.values())

score = []
for i in range(len(test_X)):
    sc_candidate = class_dict[test_X.iloc[i]['predicted_label1']] + class_dict[test_X.iloc[i]['predicted_label2']]
    sc_candidate = [x for x in sc_candidate if x in key_list]
    sc_candidate_idx = list({key: small_class_dic[key] for key in sc_candidate}.values())
    sc_candidate2 = [x for x in small_predict_prob[i] if x in sc_candidate_idx]
    first = sc_candidate2[0]
    second = sc_candidate2[1]
    label = list([key_list[val_list.index(first)], key_list[val_list.index(second)]])
    if test_y_small[i] in label :
        score.append(1)
    else :
        score.append(0)
acc = sum(score)/len(score)
print("hierachy accuracy_score", acc)
