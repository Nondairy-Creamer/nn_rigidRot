# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Dense, Input, SimpleRNN
from keras.models import Model
import numpy as np
from keras.layers import Layer
from keras import backend as K
from keras import optimizers
import keras

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces

from keras.layers import RNN
from keras.layers import SimpleRNNCell


class DynamicNonlin(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



model_in = Input([4, 2])
model_out = RnnWormNeuron(2)(model_in)
model = Model(inputs=model_in, outputs=model_out, name='my_model')

adamOpt = optimizers.Adam(lr=1e-4)
model.compile(optimizer=adamOpt, loss='mean_squared_error')

hist = model.fit(np.ones((10, 4, 2)), np.ones((10, 2))+1, verbose=2, epochs=100, batch_size=np.power(2,2))