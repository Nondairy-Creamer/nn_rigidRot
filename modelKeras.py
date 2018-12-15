from tensorflow.keras.layers import multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv2D, subtract, add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
#import tensorflow.keras as TFK
import tensorflow.keras.backend as K
import h5py as h
import numpy as np
from tensorflow.keras.layers import *


def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2):
    learning_rate = 0.001*1
    batch_size = np.power(2, 6)

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    pad_x = int((filter_shape[1] - 1) / 2)
    pad_t = int((filter_shape[0] - 1) / 2)

    input_conv = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=None))

    # left arm
    conv1 = input_conv(X_input)
    conv1_a = Activation('relu')(conv1)

    # right arm
    # make reversal layers
    reverseLayer2 = Lambda(lambda x: K.reverse(x, axes=2))

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    reversedInput = reverseLayer2(X_input)
    conv2 = reverseLayer2(input_conv(reversedInput))

    conv2_a = Activation('relu')(conv2)

    subtractedLayer = subtract([conv1_a, conv2_a])
    # subtractedLayer = concatenate([conv1_a, conv2_a], axis=3)

    # conv_x_size = int(x_layer2.shape[2])
    conv_x_size = 1
    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=None))(subtractedLayer)

    averaged_space = sum_layer(combine_filters)

    # Create model
    model = Model(inputs=X_input, outputs=averaged_space, name='SimpleMotion')

    return model, pad_x, pad_t, learning_rate, batch_size


def ln_model_deep(input_shape=(11, 9, 1), filter_shape=((21, 5), (21, 5)), num_filter=(4, 1)):
    learning_rate = 0.001*1
    batch_size = np.power(2, 6)
    # ln model of the structure
    # conv, relu, conv, relu, subtract off mirror symmetric subunits, linear combination

    # Define the input as a tensor with shape input_shape
    x_input = Input(input_shape)

    # output the amount that this model will reduce the time variable by
    pad_x = int((filter_shape[0][1] - 1) / 2 + (filter_shape[1][1] - 1) / 2)
    pad_t = int((filter_shape[0][0] - 1) / 2 + (filter_shape[1][0] - 1) / 2)

    # define the initial convolution without inputs so we can reuse it. We want to train the same weights for the mirror
    # symmetric units
    conv_1 = Conv2D(num_filter[0], filter_shape[0], strides=(1, 1), name='conv1',
                        kernel_initializer=glorot_uniform(seed=None))
    conv_2 = Conv2D(num_filter[1], filter_shape[1], strides=(1, 1), name='conv2',
                        kernel_initializer=glorot_uniform(seed=None))

    # make reversal layers to flip convolutions
    reverse_layer = Lambda(lambda lam: K.reverse(lam, axes=2))

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # START actually computing x
    # non reversed path
    x_1 = conv_1(x_input)
    x_1 = Activation('relu')(x_1)

    # space reversed convolution units
    x_1_reverse = reverse_layer(conv_1(reverse_layer(x_input)))
    x_1_reverse = Activation('relu')(x_1_reverse)

    # concatinate 2
    x_layer1 = subtract([x_1, x_1_reverse])
    # x_layer1 = concatenate([x_1, x_1_reverse], axis=3)

    # non reversed 2nd convolution
    x_2 = conv_2(x_layer1)
    x_2 = Activation('relu')(x_2)

    # space reversed second convolution
    x_2_reverse = reverse_layer(conv_2(reverse_layer(x_layer1)))
    x_2_reverse = Activation('relu')(x_2_reverse)

    # concatinate 2
    x_layer2 = concatenate([x_2, x_2_reverse], axis=3)
    #x_layer2 = subtract([x_2, x_2_reverse])
    #x_layer2 = x_2

    #conv_x_size = int(x_layer2.shape[2])
    conv_x_size = 1
    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='x_out',
               kernel_initializer=glorot_uniform(seed=None))(x_layer2)

    model_sum = sum_layer(combine_filters)

    # Create model
    model = Model(inputs=x_input, outputs=model_sum, name='DeepLn')

    return model, pad_x, pad_t, learning_rate, batch_size


def hrc_model(input_shape=(11, 9, 1), filter_shape=(21, 2), num_hrc=1):
    # set the learning rate that works for this model
    learning_rate = 0.001 * 1
    batch_size = np.power(2, 6)

    # output the amount that this model will reduce the space and time variable by
    pad_x = int((filter_shape[1] - 1) / 2)
    pad_t = int((filter_shape[0] - 1) / 2)

    # Define the input as a tensor with shape input_shape
    model_input = Input(input_shape)

    left_in = Conv2D(num_hrc, filter_shape, strides=(1, 1), name='conv1',
                       kernel_initializer=glorot_uniform(seed=None))
    right_in = Conv2D(num_hrc, filter_shape, strides=(1, 1), name='conv2',
                      kernel_initializer=glorot_uniform(seed=None))

    # make reversal layers to flip convolutions
    reverse_layer = Lambda(lambda lam: K.reverse(lam, axes=2))

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # calculate unit1 inputs
    unit1_left = left_in(model_input)
    unit1_right = right_in(model_input)

    unit1_multiply = multiply([unit1_left, unit1_right])

    # calculate unit2 inputs
    unit2_left = reverse_layer(left_in(reverse_layer(model_input)))
    unit2_right = reverse_layer(right_in(reverse_layer(model_input)))

    unit2_multiply = multiply([unit2_left, unit2_right])

    full_reich = subtract([unit1_multiply, unit2_multiply])
    # full_reich = unit1_multiply

    # combine all the correlators
    # conv_x_size = int(x_layer2.shape[2])
    conv_x_size = 1
    combine_corr = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='x_out',
                   kernel_initializer=glorot_uniform(seed=None))(full_reich)

    sum_reich = sum_layer(combine_corr)

    # Create model
    model = Model(inputs=model_input, outputs=sum_reich, name='ReichCorr')

    return model, pad_x, pad_t, learning_rate, batch_size


def load_data_rr(path):
    mat_contents = h.File(path, 'r')

    trainX = np.float32(mat_contents['trainX'].value)
    trainY = np.float32(mat_contents['trainY'].value)
    devX = np.float32(mat_contents['devX'].value)
    devY = np.float32(mat_contents['devY'].value)
    testX = np.float32(mat_contents['testX'].value)
    testY = np.float32(mat_contents['testY'].value)

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
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value
