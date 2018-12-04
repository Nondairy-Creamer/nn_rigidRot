from tensorflow.keras.layers import multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv2D, subtract, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
#import tensorflow.keras as TFK
import tensorflow.keras.backend as K
import h5py as h
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf

def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filt=2):
    learning_rate = 0.001*100

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    pad = int((filter_shape[0] - 1) / 2)

    inputConv = Conv2D(num_filt, filter_shape, strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=None))

    # left arm
    conv1 = inputConv(X_input)
    conv1_a = Activation('relu')(conv1)

    # right arm
    # make reversal layers
    reverseLayer2 = Lambda(lambda x: K.reverse(x, axes=2))

    reversedInput = reverseLayer2(X_input)
    conv2 = reverseLayer2(inputConv(reversedInput))
    conv2_a = Activation('relu')(conv2)

    subtractedLayer = subtract([conv1_a, conv2_a])
    #subtractedLayer = concatenate([conv1_a, conv2_a], axis=3)

    size_x_conv1 = int(conv1.shape[2])
    #X = Conv2D(1, (1, 1), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=None))(conv1_a)
    X = Conv2D(1, (1, size_x_conv1), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=None))(subtractedLayer)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='SimpleMotion')

    return model, pad, learning_rate


def hrc_model(input_shape=(11, 9, 1), filter_shape=(21, 2), num_filt=2):
    learning_rate = 0.001 * 1

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    pad = int((filter_shape[0] - 1) / 2)

    inputConv = Conv2D(num_filt, filter_shape, strides=(1, 1), name='conv1',
                       kernel_initializer=glorot_uniform(seed=None))

    leftArm_conv = inputConv(X_input)
    leftArm_1_conv = Lambda(lambda x: x[:, :, :, 0, None])(leftArm_conv)
    leftArm_2_conv = Lambda(lambda x: x[:, :, :, 1, None])(leftArm_conv)

    multiplyLayer_leftArm = multiply([leftArm_1_conv, leftArm_2_conv])

    reverseLayer2 = Lambda(lambda x: K.reverse(x, axes=2))
    rightArm_conv = inputConv(reverseLayer2(X_input))
    rightArm_1_conv = Lambda(lambda x: K.reverse(x[:, :, :, 0, None], axes=2))(rightArm_conv)
    rightArm_2_conv = Lambda(lambda x: K.reverse(x[:, :, :, 1, None], axes=2))(rightArm_conv)

    multiplyLayer_rightArm = multiply([rightArm_1_conv, rightArm_2_conv])

    subtractedLayer = subtract([multiplyLayer_leftArm, multiplyLayer_rightArm])

    size_x_sub = int(subtractedLayer.shape[2])
    X = Conv2D(1, (1, size_x_sub), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=None))(
        subtractedLayer)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='SimpleMotion')

    return model, pad, learning_rate


def load_data_rr(path):
    mat_contents = h.File(path, 'r')

    trainX = mat_contents['trainX'].value
    trainY = mat_contents['trainY'].value
    devX = mat_contents['devX'].value
    devY = mat_contents['devY'].value
    testX = mat_contents['testX'].value
    testY = mat_contents['testY'].value

    trainX = np.expand_dims(trainX, axis=3)
    devX = np.expand_dims(devX, axis=3)
    testX = np.expand_dims(testX, axis=3)

    trainY = np.expand_dims(trainY, axis=2)
    trainY = np.expand_dims(trainY, axis=3)
    devY = np.expand_dims(devY, axis=2)
    devY = np.expand_dims(devY, axis=3)
    testY = np.expand_dims(testY, axis=2)
    testY = np.expand_dims(testY, axis=3)

    return trainX, trainY, devX, devY, testX, testY


def r2(y_true, y_pred):
    r2_value = 1 - K.sum(K.square(y_pred - y_true)) / \
                     K.sum(np.square(y_true - K.mean(y_true)))
    return r2_value
