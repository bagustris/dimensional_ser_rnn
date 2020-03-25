# to be run on Jupyter-lab

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

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

# model1: GRU+GRU
def model1(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Bidirectional(CuDNNGRU(32, return_sequences=True))(input_speech)
    model_speech = Bidirectional(CuDNNGRU(32, return_sequences=False))(model_speech)
    model_speech = Dropout(0.3)(model_speech)

    # text Model
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    model_text = Embedding(nb_words,
                           EMBEDDING_DIM,
                           weights = [g_word_embedding_matrix],
                           trainable = True)(input_text)
    model_text = Bidirectional(CuDNNGRU(32, return_sequences=True))(model_text) #change to CuDNN if using GPU
    model_text = Bidirectional(CuDNNGRU(32, return_sequences=False))(model_text)
    model_text = Dropout(0.3)(model_text)
    
    # combination speech and text
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(16, activation='relu')(model_combined)
    model_combined = Dense(8, activation='relu')(model_combined)
    model_combined = Dropout(0.3)(model_combined)
    model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model([input_speech, input_text], model_combined)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])

    return model

model1 = model1()
model1.summary()

# train data
split = 8500
hist1 = model1.fit([voiced_feat[:split], x_train_text[:split]], vad[:split], batch_size=32, epochs=50, verbose=1, 
                  shuffle=True, validation_split=0.2)
acc1 = hist1.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc1), min(acc1), np.mean(acc1)))
# max: 77.4063, min:20.2464, avg:23.9724

loss1, mape1 = model1.evaluate([voiced_feat[split:], x_train_text[split:]], vad[split:])
print('loss: {:.2f}, mape: {:.2f}'.format(loss1, mape1))
# loss: 0.43, mape: 19.61


# model2: Dense + GRU
def model2(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Flatten()(input_speech)
    model_speech = Dense(32, activation='relu')(model_speech)
    model_speech = Dense(32)(model_speech)
    model_speech = Dropout(0.3)(model_speech)

    # text Model
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    model_text = Embedding(nb_words,
                           EMBEDDING_DIM,
                           weights = [g_word_embedding_matrix],
                           trainable = True)(input_text)
    model_text = Bidirectional(CuDNNGRU(32, return_sequences=True))(model_text) #change to CuDNN if using GPU
    model_text = Bidirectional(CuDNNGRU(32, return_sequences=False))(model_text)
    model_text = Dropout(0.3)(model_text)
    
    # combination speech and text
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(16, activation='relu')(model_combined)
    model_combined = Dense(8, activation='relu')(model_combined)
    model_combined = Dropout(0.3)(model_combined)
    model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model([input_speech, input_text], model_combined)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])

    return model

model2 = model2()
model2.summary()

# train data
split = 8500
hist2 = model2.fit([voiced_feat[:split], x_train_text[:split]], vad[:split], batch_size=32, epochs=30, verbose=1, 
                  shuffle=True, validation_split=0.2)
acc2 = hist2.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc2), min(acc2), np.mean(acc2)))
# max: 34.8221, min:19.1339, avg:21.2860

loss, mape = model2.evaluate([voiced_feat[split:], x_train_text[split:]], vad[split:])
print('loss: {:.2f}, mape: {:.2f}'.format(loss, mape))

#from keras.utils import plot_model 
#plot_model(model2,show_shapes=True, show_layer_names=False, to_file='model2_denseGru.pdf')  

# model3: Dense + GRU
def model3(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Flatten()(input_speech)
    model_speech = Dense(32, activation='relu')(model_speech)
    model_speech = Dense(32)(model_speech)
    model_speech = Dropout(0.3)(model_speech)

    # text Model
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    model_text = Embedding(nb_words,
                           EMBEDDING_DIM,
                           weights = [g_word_embedding_matrix],
                           trainable = True)(input_text)
    model_text = Bidirectional(CuDNNGRU(32, return_sequences=True))(model_text) #change to CuDNN if using GPU
    model_text = Bidirectional(CuDNNGRU(32, return_sequences=False))(model_text)
    model_text = Dropout(0.3)(model_text)
    
    # combination speech and text
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(16, activation='relu')(model_combined)
    model_combined = Dense(8, activation='relu')(model_combined)
    model_combined = Dropout(0.3)(model_combined)
    model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model([input_speech, input_text], model_combined)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])

    return model

model3 = model3()
model3.summary()

# train data
split = 8500
hist3 = model3.fit([voiced_feat[:split], x_train_text[:split]], vad[:split], batch_size=32, epochs=30, verbose=1, 
                  shuffle=True, validation_split=0.2)
acc3 = hist3.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc3), min(acc3), np.mean(acc3)))
# max: 34.8221, min:19.1339, avg:21.2860

loss3, mape3 = model3.evaluate([voiced_feat[split:], x_train_text[split:]], vad[split:])
print('loss: {:.2f}, mape: {:.2f}'.format(loss3, mape3))
# loss: 0.43, mape: 19.47
