from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.cnn_embedding_kr import CNNClassifier
from sklearn.metrics import accuracy_score
from gensim.models import word2vec, doc2vec
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import json, os
import pickle

evaluate_mode = input("학습 모드를 설정하세요 (1): ML, (2): CNN : ")

if evaluate_mode.lower() == 'cnn':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception as e:
            print(e)

    date = input("Data Dir 날짜를 입력하세요!(YYYY-mm-dd) : ")
    weight_dir = input("검정할 Data Dir를 입력하세요!(YYYY-mm-dd) : ")
    emb_mode = input("embedding model을 선택해주세요 : ")
    optimizer = input("Optimizer를 선택해주세요 (1) adam (2) radam: ")
    output_size = input("output_size 입력 (1) new_class : 43, (2) new_small : 455")
    print("embedding mode : ", emb_mode, "optimizer : ", optimizer)

    DATA_IN_PATH = './data/npy_data/{}/base_token/'.format(date)
    TRAIN_INPUT_DATA = 'train_input_bt.npy'
    TEST_INPUT_DATA = 'test_input_bt.npy'
    SEQ_DATA = 'seq_configs_bt.json'
    model_name = 'cnn_bt_embedding_{}_{}_kr'.format(emb_mode, optimizer)
    w2v_model_name = './model/w2v_model/Word2vec1(base_token).model'
    d2v_model_name = './model/doc2vec/Doc2vec1.model'

    print(model_name)
    # mode select (1) w2v (2) d2v
    sequence_data =json.load(open(DATA_IN_PATH + SEQ_DATA, 'r'))
    text_data = sequence_data['total_text_list']

    if emb_mode == 'w2v':
        word_vectorizer = word2vec.Word2Vec.load(w2v_model_name)
        word_vectorizer.build_vocab(text_data, update=True)
        print(str(word_vectorizer))
    elif emb_mode == 'd2v':
        doc_vectorizer = doc2vec.Doc2Vec.load(d2v_model_name)
    else:
        print("(1) w2v, (2) d2v 중에서 고르시오")
        raise ValueError

    DATA_OUT_PATH = './model/cnn_model/{}/'.format(weight_dir)
    TRAIN_LABEL_DATA = 'train_label.npy'
    TEST_LABEL_DATA = 'test_label.npy'
    DATA_CONFIGS = 'data_configs.json'

    train_X = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
    train_Y = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))
    test_X = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
    test_Y = np.load(open(DATA_IN_PATH + TEST_LABEL_DATA, 'rb'))
    data_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
    data_configs['vocab_size']

    save_model_name = 'cnn_classifier_model({}-{})'.format(emb_mode, optimizer)
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    VALID_SPLIT = 0.2
    MAX_LEN = train_X.shape[1]
    word_index = data_configs['vocab']

    EMBEDDING_DIM=300
    vocabulary_size= len(word_index)+1
    NUM_WORDS = len(word_index)+1
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

    j, k = 0, 0
    for word, i in word_index.items():
        if i>=NUM_WORDS:
            print("Index Error")
            continue
        try:
            if j == 0:
                print("{} train Start".format(emb_mode))
            if emb_mode == 'w2v':
                embedding_vector = word_vectorizer[word]
                embedding_matrix[i] = embedding_vector
            elif emb_mode == 'd2v':
                embedding_vector = doc_vectorizer.infer_vector([word])
                embedding_matrix[i] = embedding_vector
            else:
                print("(1) w2v, (2) d2v 중에서 고르시오")
                raise ValueError
            j += 1
        except KeyError as e:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
            k += 1

    print("embedding matrix dim : ", embedding_matrix.shape)

    kargs = {'vocab_size': data_configs['vocab_size']+1,
            'embedding_size': 300,
            'embedding_matrix':embedding_matrix,
            'num_filters': 128,
            'dropout_rate': 0.5,
            'hidden_dimension': 1000,
            'output_dimension':39,
            'model_name':model_name}

    model = CNNClassifier(**kargs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=10)
    model.build(train_X.shape)
    print(model.summary())

    SAVE_FILE_NM = 'weights.h5' #저장된 best model 이름
    model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))
    print(model.evaluate(test_X, test_Y))

    def return_two_calss_acc(model, x_test, y_test):
        y_pred_prob = model.predict(x_test)
        L = np.argsort(-y_pred_prob, axis=1)
        two_pred = L[:,0:2]
        score = []
        score = []
        for i in range(len(y_test)):
            label = two_pred[i]
            if y_test[i] in label :
                score.append(1)
            else :
                score.append(0)
        acc = sum(score)/len(y_test)
        return acc

    two_acc = return_two_calss_acc(model, test_X, test_Y)
    print("CNN classifier 2class 정확도: {:.3f}".format(two_acc))
else:
    def return_two_calss_acc(clf, x_test, y_test):
        y_pred_prob = clf.predict_proba(X_test)
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

    DATA_IN_PATH = './data/ml_data/'
    TEST_INPUT_DATA = 'test_input.npy'
    TEST_LABEL = 'test_label.npy'

    X_test = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
    y_test = np.load(open(DATA_IN_PATH + TEST_LABEL, 'rb'))

    filename = './model/ml_model/LR_clf.sav'
    LR_clf = pickle.load(open(filename, 'rb'))
    y_pred = LR_clf.predict(X_test)
    print("Logistics Regression 1class 정확도: {:.3f}".format(accuracy_score(y_pred, y_test)))
    two_acc = return_two_calss_acc(LR_clf, X_test, y_test)
    print("Logistics Regression 2class 정확도: {:.3f}".format(two_acc))

    filename = './model/ml_model/mlp_clf.sav'
    MLP_clf = pickle.load(open(filename, 'rb'))
    y_pred = MLP_clf.predict(X_test)
    print("MLP 1class 정확도: {:.3f}".format(accuracy_score(y_pred, y_test)))
    two_acc = return_two_calss_acc(MLP_clf, X_test, y_test)
    print("MLP 2class 정확도: {:.3f}".format(two_acc))

    filename = './model/ml_model/sgd_clf.sav'
    SGD_clf = pickle.load(open(filename, 'rb'))
    y_pred = SGD_clf.predict(X_test)
    print("SGD classifier 1class 정확도: {:.3f}".format(accuracy_score(y_pred, y_test)))
    two_acc = return_two_calss_acc(MLP_clf, X_test, y_test)
    print("SGD classifie 2class 정확도: {:.3f}".format(two_acc))
