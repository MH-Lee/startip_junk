import pandas as pd
import numpy as np
import re, os, json, sys
import tensorflow as tf
import argparse
import datetime
from ast import literal_eval
from matplotlib import rc, rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
sys.path.append(os.pardir)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)

total_data = pd.read_excel('./data/doc_set_final_version3.xlsx')
stop_words = pd.read_csv('./data/korean_stopwords.txt')['stopwords'].tolist()
# total_data['token'] =  total_data['token'].apply(lambda x: literal_eval(x))
total_data = total_data.sample(frac=1, random_state=1234)

## 기존 토큰 활용
data = total_data[['token', 'new_small_class']]
token_list = total_data['token'].tolist()
target = total_data['new_class'].tolist()

train_x_data, test_x_data, train_y, test_y = train_test_split(data, target,
                                                        test_size=0.3,
                                                        stratify=target,
                                                        shuffle=True,
                                                        random_state=1234)

print(len(train_x), len(test_x), len(train_y), len(test_y))
print(len(train_x[0]), len(test_x[0]), len(np.unique(train_y)), len(np.unique(test_y)))

total_target = train_y + test_y
total_small_target = train_x_data.new_small_class.tolist() + test_x_data.new_small_class.tolist()
train_x_data = train_x_data.token.tolist()
test_x_data = test_x_data.token.tolist()

LABEL_JSON_NAME = './label_data/label.json'
LABEL_JSON_NAME_SMALL = './label_data/label_small.json'

## new_class label encoder
lbl_e = LabelEncoder()
target_label = lbl_e.fit_transform(total_target)
le_name_mapping = dict(zip(lbl_e.transform(lbl_e.classes_), lbl_e.classes_))
le_dict = dict()
for k, v in le_name_mapping.items():
    le_dict[str(k)] = v
json.dump(le_dict, open(LABEL_JSON_NAME, 'w'), ensure_ascii=True)

train_labels = target_label[:len(train_x_data)]
test_labels = target_label[len(train_x_data):]

print("클래스 개수 : ", label_number)
print("학습데이터 클래스 개수 : ", len(np.unique(train_labels)))
print("검증데이터 클래스 개수 : ",len(np.unique(test_labels)))
train_y_s = tf.keras.utils.to_categorical(train_labels, num_classes = label_number)
test_y_s= tf.keras.utils.to_categorical(test_labels, num_classes = label_number)

## new_small_class label encoder

lbl_e2 = LabelEncoder()
target_label_small = lbl_e2.fit_transform(total_small_target)
le_small_name_mapping = dict(zip(lbl_e2.transform(lbl_e2.classes_), lbl_e2.classes_))
le_dict2 = dict()
for k, v in le_small_name_mapping.items():
    le_dict2[str(k)] = v

json.dump(le_dict2, open(LABEL_JSON_NAME_SMALL, 'w'), ensure_ascii=True)

train_small_labels = target_label_small[:len(train_x_data)]
test_small_labels = target_label_small[len(train_x_data):]

print("small 클래스 개수 : ", label_number_small)
print("학습데이터 클래스 개수 : ", len(np.unique(train_small_labels)))
print("검증데이터 클래스 개수 : ",len(np.unique(test_small_labels)))

train_y_ss = tf.keras.utils.to_categorical(train_small_labels, num_classes = label_number_small)
test_y_ss= tf.keras.utils.to_categorical(test_small_labels, num_classes = label_number_small)

after_len = [len(word) for word in train_x]
print('전처리 후 명사 길이 최대 값: {}'.format(np.max(after_len)))
print('전처리 후 명 길이 최소 값: {}'.format(np.min(after_len)))
print('전처리 후 명 길이 평균 값: {:.2f}'.format(np.mean(after_len)))
print('전처리 후 명 길이 표준편차: {:.2f}'.format(np.std(after_len)))
print('전처리 후 명 길이 중간 값: {}'.format(np.median(after_len)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('전처리 후 명 길이 제 1 사분위: {}'.format(np.percentile(after_len, 25)))
print('전처리 후 명 길이 제 3 사분위: {}'.format(np.percentile(after_len, 75)))

tokenizer = Tokenizer(split=',', filters="!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
tokenizer.fit_on_texts(token_list)
train_sequence = tokenizer.texts_to_sequences(train_x_data)
test_sequence = tokenizer.texts_to_sequences(test_x_data)

sequence_data = dict()
sequence_data['train_seq'] = train_sequence
sequence_data['test_seq'] = test_sequence
sequence_data['total_text_list'] = token_list
sequence_data['configs'] = tokenizer.get_config()

word_idx = tokenizer.word_index
MAX_SEQUENCE_LENGTH = int(np.median(after_len))

DATA_OUT_PATH = './data/npy_data/{}/{}/'.format(Today, 'base_token')

## Make output save directory
if os.path.exists(DATA_OUT_PATH):
    print("{} -- Folder already exists \n".format(DATA_OUT_PATH))
else:
    os.makedirs(DATA_OUT_PATH, exist_ok=True)
    print("{} -- Folder create complete \n".format(DATA_OUT_PATH))

train_input = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(train_labels)
train_small_labels = np.array(train_small_labels)
test_input = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_labels = np.array(test_labels)
test_small_labels = np.array(test_small_labels)

# Data save label
TRAIN_INPUT_DATA = 'train_input_bt.npy'
TEST_INPUT_DATA = 'test_input_bt.npy'
DATA_CONFIGS = 'data_configs.json'
SEQ_CONFIGS = 'seq_configs_bt.json'

# Train label save file name
TRAIN_LABEL = 'train_label.npy'
TRAIN_LABEL_SPARSE = 'train_label_sparse.npy'
TRAIN_LABEL_SMALL = 'train_label_small.npy'
TRAIN_LABEL_SMALL_SPARSE = 'train_label_small_sparse.npy'

# Test label save file name
TEST_LABEL = 'test_label.npy'
TEST_LABEL_SPARSE = 'test_label_sparse.npy'
TEST_LABEL_SMALL = 'test_label_small.npy'
TEST_LABEL_SMALL_SPARSE = 'test_label_small_sparse.npy'

data_configs = {}
data_configs['vocab'] = word_idx
data_configs['vocab_size'] = len(word_idx)

# 전처리 된 데이터를 넘파이 형태로 저장
np.save(open(DATA_OUT_PATH + TRAIN_INPUT_DATA, 'wb'), train_input)
np.save(open(DATA_OUT_PATH + TEST_INPUT_DATA, 'wb'), test_input)

# save label numpy file
np.save(open(DATA_OUT_PATH + TRAIN_LABEL, 'wb'), train_labels)
np.save(open(DATA_OUT_PATH + TEST_LABEL, 'wb'), test_labels)
np.save(open(DATA_OUT_PATH + TRAIN_LABEL_SPARSE, 'wb'), train_y_s)
np.save(open(DATA_OUT_PATH + TEST_LABEL_SPARSE, 'wb'), test_y_s)

# save small label numpy file
np.save(open(DATA_OUT_PATH + TRAIN_LABEL_SMALL, 'wb'), train_small_labels)
np.save(open(DATA_OUT_PATH + TEST_LABEL_SMALL, 'wb'), test_small_labels)
np.save(open(DATA_OUT_PATH + TRAIN_LABEL_SMALL_SPARSE, 'wb'), train_y_ss)
np.save(open(DATA_OUT_PATH + TEST_LABEL_SMALL_SPARSE, 'wb'), test_y_ss)

json.dump(data_configs, open(DATA_OUT_PATH + DATA_CONFIGS, 'w'), ensure_ascii=True)
json.dump(sequence_data, open(DATA_OUT_PATH + SEQ_CONFIGS, 'w'), ensure_ascii=True)
