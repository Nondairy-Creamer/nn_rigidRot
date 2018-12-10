from tensorflow.keras.layers import multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv2D, subtract, add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
#import tensorflow.keras as TFK
import tensorflow.keras.backend as K
import h5py as h
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf


def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2):
    learning_rate = 0.001*1
    batch_size = np.power(2, 6)

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    pad = int((filter_shape[0] - 1) / 2)

    inputConv = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=None))

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
    subtractedLayer_with_lin = concatenate([subtractedLayer, conv1, conv2], axis=3)
    #subtractedLayer_with_lin = conv1_a
    #subtractedLayer_with_lin = subtractedLayer

    size_x_conv1 = int(subtractedLayer_with_lin.shape[2])
    X = Conv2D(1, (1, size_x_conv1), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=None))(subtractedLayer_with_lin)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='SimpleMotion')

    return model, pad, learning_rate, batch_size


def ln_model_deep(input_shape=(11, 9, 1), filter_shape=((21, 5), (21, 5)), num_filter=(4, 4)):
    learning_rate = 0.001*1
    batch_size = np.power(2, 8)
    # ln model of the structure
    # conv, relu, conv, relu, subtract off mirror symmetric subunits, linear combination

    # Define the input as a tensor with shape input_shape
    x_input = Input(input_shape)

    # output the amount that this model will reduce the time variable by
    pad = int((filter_shape[0][0] - 1) / 2 + (filter_shape[1][0] - 1) / 2)

    # define the initial convolution without inputs so we can reuse it. We want to train the same weights for the mirror
    # symmetric units
    conv_1 = Conv2D(num_filter[0], filter_shape[0], strides=(1, 1), name='conv1',
                        kernel_initializer=glorot_uniform(seed=None))
    conv_2 = Conv2D(num_filter[1], filter_shape[1], strides=(1, 1), name='conv2',
                        kernel_initializer=glorot_uniform(seed=None))
    # make reversal layers to flip convolutions
    reverse_layer = Lambda(lambda lam: K.reverse(lam, axes=2))

    # START actually computing x
    # non reversed path
    x_1 = conv_1(x_input)
    x_1 = Activation('relu')(x_1)

    # space reversed convolution units
    x_1_reverse = reverse_layer(conv_1(reverse_layer(x_input)))
    x_1_reverse = Activation('relu')(x_1_reverse)

    # concatinate 2
    x_layer1 = concatenate([x_1, x_1_reverse], axis=3)
    #x_layer1 = x_1

    # non reversed 2nd convolution
    x_2 = conv_2(x_layer1)
    x_2 = Activation('relu')(x_2)

    # space reversed second convolution
    x_2_reverse = reverse_layer(conv_2(reverse_layer(x_layer1)))
    x_2_reverse = Activation('relu')(x_2_reverse)

    # concatinate 2
    x_layer2 = concatenate([x_2, x_2_reverse], axis=3)
    #x_layer2 = x_2

    size_x_layer2 = int(x_layer2.shape[2])
    x_out = Conv2D(1, (1, size_x_layer2), strides=(1, 1), name='x_out',
               kernel_initializer=glorot_uniform(seed=None))(x_layer2)

    # Create model
    model = Model(inputs=x_input, outputs=x_out, name='DeepLn')

    return model, pad, learning_rate, batch_size


def hrc_model(input_shape=(11, 9, 1), filter_shape=(21, 2), num_filter=2):
    # set the learning rate that works for this model
    learning_rate = 0.001 * 1

    # output the amount that this model will reduce the time variable by
    pad = int((filter_shape[0] - 1) / 2)+int((filter_shape[0] - 1) / 2)


    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)



    inputConv = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1',
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
