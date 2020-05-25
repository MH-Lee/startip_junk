from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.cnn_embedding_kr import CNNClassifier
from gensim.models import word2vec, doc2vec, FastText
import tensorflow as tf
import numpy as np
import json, os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)

w2v_model_name = './model/word_embedding/Word2vec1(base_token).model'
word_vectorizer = word2vec.Word2Vec.load(w2v_model_name)
model_name = 'cnn_bt_embedding_kr'

DATA_IN_PATH = './data/npy_data/2020-04-25/base_token/'
DATA_OUT_PATH = './model/cnn_model/'
TRAIN_INPUT_DATA = 'train_input_bt.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TEST_INPUT_DATA = 'test_input_bt.npy'
TEST_LABEL_DATA = 'test_label.npy'
DATA_CONFIGS = 'data_configs.json'

train_X = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
train_Y = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))
test_X = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
test_Y = np.load(open(DATA_IN_PATH + TEST_LABEL_DATA, 'rb'))
data_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
data_configs['vocab_size']

BATCH_SIZE = 200
NUM_EPOCHS = 100
VALID_SPLIT = 0.2
MAX_LEN = train_X.shape[1]
word_index = data_configs['vocab']

EMBEDDING_DIM=300
vocabulary_size= len(word_index)+1
NUM_WORDS = len(word_index)+1
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
j = 0
k = 0
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        j += 1
        embedding_vector = word_vectorizer[word]
        embedding_matrix[i] = embedding_vector
    except KeyError as e:
        k += 1
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

print(j, k)

kargs = {'vocab_size': data_configs['vocab_size']+1,
        'embedding_size': 300,
        'embedding_matrix':embedding_matrix,
        'num_filters': 128,
        'dropout_rate': 0.5,
        'hidden_dimension': 1000,
        'output_dimension':43,
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


cp_callback = ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True)

history = model.fit(train_X, train_Y,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_split=VALID_SPLIT,
                    callbacks=[earlystop_callback, cp_callback])

model.plot_graphs(history, 'accuracy')
model.plot_graphs(history, 'loss')

# # 결과 평가하기
SAVE_FILE_NM = 'weights.h5' #저장된 best model 이름
model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))
model.evaluate(test_X, test_Y)
