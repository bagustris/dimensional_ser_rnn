# apsipa 2019: dimensional speech emotion recognition from acoustic feature and word embedding

import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU
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

# load text
# Word embedding calculation
MAX_SEQUENCE_LENGTH = 500

text = [t['transcription'] for t in data]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_path = '/media/bagus/data01/dataset/fasttext/' #crawl-300d-2M-subword/'
#file_loc = data_path + 'crawl-300d-2M-subword.vec'
file_loc = data_path + 'wiki-news-300d-1M-subword.vec'
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

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

# model2: Dense + GRU
def model2(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Flatten()(input_speech)
    model_speech = Dense(1024, activation='relu')(model_speech)
    model_speech = Dense(512)(model_speech)
    model_speech = Dropout(0.3)(model_speech)

    # text Model
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    model_text = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(input_text)
    model_text = Bidirectional(CuDNNGRU(64, return_sequences=True))(model_text) #change to CuDNN if using GPU
    model_text = Bidirectional(CuDNNGRU(64, return_sequences=False))(model_text)
    model_text = Dropout(0.3)(model_text)
    
    # combination speech and text
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(128, activation='relu')(model_combined)
    model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model([input_speech, input_text], model_combined)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])

    return model

model2 = model2()
model2.summary()

# train data
hist2 = model2.fit([voiced_feat, x_train_text], vad, batch_size=32, epochs=30, verbose=1, 
                  shuffle=True, validation_split=0.2)
acc2 = hist2.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc2), min(acc2), np.mean(acc2)))
# max: 24.7440, min:18.9866, avg:21.0513
# max: 31.2443, min:18.6922, avg:19.7895