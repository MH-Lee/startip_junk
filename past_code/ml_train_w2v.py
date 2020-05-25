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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
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

w2v_model_name = './model/word_embedding/Word2vec1(base_token).model'
word_vectorizer = Word2Vec.load(w2v_model_name)
word_vectorizer.wv.vectors.shape
word_index = word_vectorizer.wv.index2word
EMBEDDING_DIM = word_vectorizer.trainables.layer1_size

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
train_X = train_X.token.tolist()
test_y_small = test_X.new_small_class.tolist()
test_X = test_X.token.tolist()

train_doc_weight = list()
for word_list in train_X:
    mean_w2v = np.mean([word_vectorizer.wv[w] for w in word_list if w in word_index] or
                        [np.random.normal(0,np.sqrt(0.25), EMBEDDING_DIM)], axis=0)
    train_doc_weight.append(mean_w2v)


test_doc_weight = list()
for word_list in test_X:
    mean_w2v = np.mean([word_vectorizer.wv[w] for w in word_list if w in word_index] or
                        [np.random.normal(0,np.sqrt(0.25), EMBEDDING_DIM)], axis=0)
    test_doc_weight.append(mean_w2v)


clf = LogisticRegression(
    solver='newton-cg',
    random_state=1234)

clf.fit(train_doc_weight, train_y)
filename = './model/ml_model/LR_clf_w2v_mean.sav'
pickle.dump(clf, open(filename, 'wb'))

LR_scores = cross_val_score(clf, train_doc_weight, train_y, cv=10)
print("Logistics Regression CV Accuracy", LR_scores)
print("Logistics Regression CV 평균 Accuracy: ", np.mean(LR_scores))
print("Logistics Regression CV Std: ", np.std(LR_scores))

y_pred = clf.predict(test_doc_weight)
print("Logistic Regression 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y)))
print("Logistic Regression 1class Recall: {:.4f}".format(recall_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class precision_score: {:.4f}".format(precision_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class f1_score: {:.4f}".format(f1_score(test_y, y_pred, average='weighted')))
two_acc_small = return_two_calss_acc(clf, test_doc_weight, test_y)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))

clf.fit(train_doc_weight, train_y_small)
filename = './model/ml_model/LR_clf_small_w2v.sav'
pickle.dump(clf, open(filename, 'wb'))

# # LR_scores_small = cross_val_score(clf, X_train, train_y_small, cv=10)
# print("Logistics Regression CV small class Accuracy", LR_scores_small)
# print("Logistics Regression CV small class 평균 Accuracy: ", np.mean(LR_scores_small))
# print("Logistics Regression CV small class Std: ", np.std(LR_scores_small))

y_pred = clf.predict(test_doc_weight)
print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred, average='weighted')))

two_acc_small = return_two_calss_acc(clf, test_doc_weight, test_y_small)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))
#
# ### MLP
# mlp_clf = MLPClassifier(hidden_layer_sizes=(1000,),
#                         max_iter=100,
#                         alpha=1e-4,
#                         solver='adam',
#                         verbose=0,
#                         tol=1e-4,
#                         early_stopping=True,
#                         random_state=1234,
#                         shuffle=True,
#                         learning_rate='adaptive',
#                         learning_rate_init=.01)
#
# MLP_scores = cross_val_score(mlp_clf, train_doc_weight, train_y, cv=10)
# print("MLP CV Accuracy", MLP_scores)
# print("MLP CV 평균 Accuracy: ", np.mean(MLP_scores))
#
# mlp_clf.fit(train_doc_weight, train_y)
# filename = './model/ml_model/mlp_clf_w2v.sav'
# pickle.dump(mlp_clf, open(filename, 'wb'))
#
# y_pred = mlp_clf.predict(test_doc_weight)
# print("MLP 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y)))
# print("MLP 1class Recall: {:.4f}".format(recall_score(test_y, y_pred, average='weighted')))
# print("MLP 1class precision_score: {:.4f}".format(precision_score(test_y, y_pred, average='weighted')))
# print("MLP 1class f1_score: {:.4f}".format(f1_score(test_y, y_pred, average='weighted')))
# two_acc = return_two_calss_acc(mlp_clf, X_test, y_test)
# print("MLP 2class 정확도: {:.3f}".format(two_acc))
# print()
# print()
#
# mlp_clf.fit(train_doc_weight, train_y_small)
# filename = './model/ml_model/mlp_clf_small_w2v.sav'
# pickle.dump(clf, open(filename, 'wb'))
#
# y_pred_small = mlp_clf.predict(test_doc_weight)
# print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred_small, test_y_small)))
# print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred_small, average='weighted')))
# print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred_small, average='weighted')))
# print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred_small, average='weighted')))
#
# two_acc_small = return_two_calss_acc(clf, test_doc_weight, test_y_small)
# print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))
