import modelKeras as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import scipy.io as sio
import tensorflow as tf
import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.layers import multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv1D, subtract, add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


out = np.zeros((1, 5, 1))
out[0, 0, 0] = 0
out[0, 1, 0] = 0
out[0, 2, 0] = 0
out[0, 3, 0] = 0
out[0, 4, 0] = 1

data = np.zeros((1, 5, 1))
data[0, 0, 0] = 0
data[0, 1, 0] = 0
data[0, 2, 0] = 0
data[0, 3, 0] = 0
data[0, 4, 0] = 1


model_in = Input(shape=(5, 1))
scalar = K.variable(2)
model_out = Lambda(lambda lam: lam[0]*scalar)([model_in])

model = Model(inputs=model_in, outputs=model_out, name='ln_model')

# set up the model and fit it
adamOpt = optimizers.Adam(lr=1, decay=0)
model.compile(optimizer=adamOpt, loss='mean_squared_error')
hist = model.fit(data, out, verbose=2, epochs=1000, batch_size=1)

# model.save_weights('G:\\My Drive\\keras_model.h5')
model.save('G:\\My Drive\\keras_model.h5')

a = 1
