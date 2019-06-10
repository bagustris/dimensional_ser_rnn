# cocosda 2019: dimensional speech emotion recognition from acoustic feature and word embedding
# to be run on jupyter-lab, block certain lines, Shift-Enter to execute

# uncomment for running on CPU
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
split = 8500

# model1
def speech_model4(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Bidirectional(GRU(64, return_sequences=True))(input_speech)
    model_speech = Bidirectional(AttentionDecoder(32,32))(model_speech)
    model_speech = Flatten()(model_speech)
    model_speech = Dropout(0.3)(model_speech)
    model_speech = Dense(3, activation='linear')(model_speech)
    
    model = Model(inputs=input_speech, outputs=model_speech)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape', 'mae'])
    return model

model4 = speech_model4()
hist4 = model4.fit(voiced_feat[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
acc4 = hist4.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc4), min(acc4), np.mean(acc4)))

# evaluation 
eval_metrik4 = model4.evaluate(voiced_feat[split:], vad[split:])
print(eval_metrik4)

