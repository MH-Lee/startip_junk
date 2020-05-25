from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.cnn_kr import CNNClassifier
import tensorflow as tf
import numpy as np
import json, os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)

inputs = input("학습 모드를 설정하세요 (1): bt, (2): nouns, (3)total")
print("학습 모드 : ", inputs)
if inputs == 'bt':
    DATA_IN_PATH = './data/npy_data/base_token/'
    TRAIN_INPUT_DATA = 'train_input_bt.npy'
    TEST_INPUT_DATA = 'test_input_bt.npy'
    model_name = 'cnn_bt_kr'
elif inputs == 'nouns':
    DATA_IN_PATH = './data/npy_data/nouns/'
    TRAIN_INPUT_DATA = 'train_input_nouns.npy'
    TEST_INPUT_DATA = 'test_input_nouns.npy'
    model_name = 'cnn_nouns_kr'
elif inputs == 'total':
    DATA_IN_PATH = './data/npy_data/total/'
    TRAIN_INPUT_DATA = 'train_input_total.npy'
    TEST_INPUT_DATA = 'test_input_total.npy'
    model_name = 'cnn_total_kr'
else:
    print("(1) ~ (3)번중에 고르세요")

DATA_OUT_PATH = './model/cnn_model/'
TRAIN_LABEL_DATA = 'train_label.npy'
TEST_LABEL_DATA = 'test_label.npy'
DATA_CONFIGS = 'data_configs.json'

train_X = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
train_Y = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))
test_X = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
test_Y = np.load(open(DATA_IN_PATH + TEST_LABEL_DATA, 'rb'))
data_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
print("vacab_size : ", data_configs['vocab_size'])

BATCH_SIZE = 512
NUM_EPOCHS = 1000
VALID_SPLIT = 0.2
MAX_LEN = train_X.shape[1]

kargs = {'vocab_size': data_configs['vocab_size']+1,
        'embedding_size': 300,
        'num_filters': 128,
        'dropout_rate': 0.5,
        'hidden_dimension': 512,
        'output_dimension':40,
        'model_name':model_name}

model = CNNClassifier(**kargs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=10)

checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))


cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

model.plot_graphs(history, 'accuracy')
model.plot_graphs(history, 'loss')

# # 결과 평가하기
SAVE_FILE_NM = 'weights.h5' #저장된 best model 이름

model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))
model.evaluate(test_X, test_Y)
