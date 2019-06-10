# OCOCOSDA 2019: dimensional speech emotion recognition from text feature

# uncomment these to run on CPU only
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, LSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU, GRU
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from features import *
from helper import *
from attention_helper import *

np.random.seed(99)

# loat output/label data
path = '/media/bagus/data01/dataset/IEMOCAP_full_release/data_collected_full.pickle'

with open(path, 'rb') as handle:
    data = pickle.load(handle)
len(data)

v = [v['v'] for v in data]
a = [a['a'] for a in data]
d = [d['d'] for d in data]

vad = np.array([v, a, d])
vad = vad.T
print(vad.shape)

# load input (speech feature) data
voiced_feat = np.load('voiced_feat_full.npy')
print(voiced_feat.shape)

text = [t['transcription'] for t in data]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

MAX_SEQUENCE_LENGTH = len(max(text, key=len))
x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_path = '/media/bagus/data01/github/IEMOCAP-Emotion-Detection/data/'
file_loc = data_path + 'glove.840B.300d.txt'
print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding

f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) + 1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

# split train/test
split = 8500

# model: GRU
def text_model():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    #net = Embedding(2737, 128, input_length=500)(inputs)
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                  trainable = True)(inputs)
    net = Bidirectional(GRU(32, return_sequences=True))(net)
    net = Bidirectional(AttentionDecoder(32,32))(net)
    net = Flatten()(net)
    net = Dropout(0.3)(net)
    net = Dense(3)(net) #linear activation
    model = Model(inputs=inputs, outputs=net) #[out1, out2, out3]
    model.compile(optimizer='rmsprop', loss='mse', metrics= ['mape', 'mae'])
    
    return model

model = text_model()
hist = model.fit(x_train_text[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
acc = hist.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc), min(acc), np.mean(acc)))

# evaluation 
eval_metrik = model.evaluate(x_train_text[split:], vad[split:])
print(eval_metrik)
