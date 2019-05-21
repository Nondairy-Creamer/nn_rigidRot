from tensorflow.keras.layers import multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv2D, subtract, add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
#import tensorflow.keras as TFK
import tensorflow.keras.backend as K
import h5py as h
import numpy as np
from tensorflow.keras.layers import *
import scipy.io as sio
from tensorflow.keras import regularizers

def l1_reg_sqrt(weight_matrix):
    return 0.1 * K.sum(K.sqrt(K.abs(weight_matrix)))

def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
    learning_rate = 0.001*100
    batch_size = np.power(2, 6)
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))

    conv1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(image_in)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # conv_x_size = int(x_layer2.shape[2])
    if sum_over_space:
        conv_x_size = conv1.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(conv1)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='ln_model')

    return model, pad_x, pad_t, learning_rate, batch_size


def ln_model_medulla(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
    learning_rate = 0.001*100
    batch_size = np.power(2, 6)
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))*2

    conv1 = Conv2D(4, [filter_shape[0], 1], strides=(1, 1), name='conv1_1',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(image_in)

    conv1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1_2',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(conv1)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # conv_x_size = int(x_layer2.shape[2])
    if sum_over_space:
        conv_x_size = conv1.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(conv1)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='ln_model_medulla')

    return model, pad_x, pad_t, learning_rate, batch_size


def ln_model_flip(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
    learning_rate = 0.001*100
    batch_size = np.power(2, 6)
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))

    input_conv = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)

    conv1 = input_conv(image_in)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))
    reverseLayer2 = Lambda(lambda x: K.reverse(x, axes=2))

    reversedInput = reverseLayer2(image_in)
    conv2 = reverseLayer2(input_conv(reversedInput))

    subtractedLayer = subtract([conv1, conv2])

    # conv_x_size = int(x_layer2.shape[2])
    if sum_over_space:
        conv_x_size = conv1.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(subtractedLayer)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='ln_model_flip')

    return model, pad_x, pad_t, learning_rate, batch_size


def ln_model_deep(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=(2, 2), sum_over_space=True):
    learning_rate = 0.001 * 10
    batch_size = np.power(2, 8)
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))*2

    conv_l1 = Conv2D(num_filter[0], (filter_shape[0], filter_shape[1]*0+1), strides=(1, 1), name='L1',
                     kernel_initializer=glorot_uniform(seed=None),
                     activation='relu',
                     kernel_regularizer=l1_reg_sqrt)(image_in)
    conv1 = Conv2D(num_filter[1], filter_shape, strides=(1, 1), name='T4T5',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(conv_l1)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # conv_x_size = int(x_layer2.shape[2])
    if sum_over_space:
        conv_x_size = conv1.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(conv1)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='ln_model_deep')

    return model, pad_x, pad_t, learning_rate, batch_size


def ln_model_deep_2(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=(2, 2), sum_over_space=True):
    learning_rate = 0.001 * .1
    batch_size = np.power(2, 6)
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1))*4
    pad_t = int((filter_shape[0] - 1))*4

    conv_l1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='L1',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(image_in)
    conv_l1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='L2',
                     kernel_initializer=glorot_uniform(seed=None),
                     activation='relu',
                     kernel_regularizer=l1_reg_sqrt)(conv_l1)
    conv_l1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='L3',
                     kernel_initializer=glorot_uniform(seed=None),
                     activation='relu',
                     kernel_regularizer=l1_reg_sqrt)(conv_l1)
    conv1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='T4T5',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(conv_l1)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # conv_x_size = int(x_layer2.shape[2])
    if sum_over_space:
        conv_x_size = conv1.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(conv1)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='ln_model_deep_2')

    return model, pad_x, pad_t, learning_rate, batch_size


def hrc_model(input_shape=(11, 9, 1), filter_shape=(21, 2), num_hrc=1, sum_over_space=True):
    # set the learning rate that works for this model
    learning_rate = 0.001 * 1
    batch_size = np.power(2, 6)
    reg_val = 0.1

    # output the amount that this model will reduce the space and time variable by
    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))

    # Define the input as a tensor with shape input_shape
    model_input = Input(input_shape)

    left_in = Conv2D(num_hrc, filter_shape, strides=(1, 1), name='conv1',
                     kernel_initializer=glorot_uniform(seed=None),
                     kernel_regularizer=regularizers.l1(reg_val),
                     use_bias=False)(model_input)
    right_in = Conv2D(num_hrc, filter_shape, strides=(1, 1), name='conv2',
                      kernel_initializer=glorot_uniform(seed=None),
                      kernel_regularizer=regularizers.l1(reg_val),
                      use_bias=False)(model_input)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    multiply_layer = multiply([left_in, right_in])

    # full_reich = unit1_multiply

    # combine all the correlators
    if sum_over_space:
        conv_x_size = multiply_layer.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_corr = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='x_out',
                          kernel_initializer=glorot_uniform(seed=None),
                          kernel_regularizer=regularizers.l1(reg_val),
                          use_bias=False)(multiply_layer)

    # Create model
    model = Model(inputs=model_input, outputs=combine_corr, name='hrc_model')

    return model, pad_x, pad_t, learning_rate, batch_size


def hrc_model_sep(input_shape=(11, 9, 1), filter_shape=(21, 2), num_hrc=1, sum_over_space=True):
    # set the learning rate that works for this model
    learning_rate = 0.001 * 1
    batch_size = np.power(2, 6)
    reg_val = 0.1

    # output the amount that this model will reduce the space and time variable by
    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))

    # Define the input as a tensor with shape input_shape
    model_input = Input(input_shape)

    left_in_t = Conv2D(num_hrc, (filter_shape[0], 1), strides=(1, 1), name='conv1_t',
                     kernel_initializer=glorot_uniform(seed=None),
                     kernel_regularizer=regularizers.l1(reg_val),
                     use_bias=False)(model_input)
    left_in_xt = Conv2D(num_hrc, (1, filter_shape[1]), strides=(1, 1), name='conv1_xt',
                     kernel_initializer=glorot_uniform(seed=None),
                     kernel_regularizer=regularizers.l1(reg_val),
                     use_bias=False)(left_in_t)

    right_in_t = Conv2D(num_hrc, (filter_shape[0], 1), strides=(1, 1), name='conv2_t',
                      kernel_initializer=glorot_uniform(seed=None),
                      kernel_regularizer=regularizers.l1(reg_val),
                      use_bias=False)(model_input)
    right_in_xt = Conv2D(num_hrc, (1, filter_shape[1]), strides=(1, 1), name='conv2_xt',
                      kernel_initializer=glorot_uniform(seed=None),
                      kernel_regularizer=regularizers.l1(reg_val),
                      use_bias=False)(right_in_t)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    multiply_layer = multiply([left_in_xt, right_in_xt])

    # full_reich = unit1_multiply

    # combine all the correlators
    if sum_over_space:
        conv_x_size = multiply_layer.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_corr = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='x_out',
                          kernel_initializer=glorot_uniform(seed=None),
                          kernel_regularizer=regularizers.l1(reg_val),
                          use_bias=False)(multiply_layer)

    # Create model
    model = Model(inputs=model_input, outputs=combine_corr, name='hrc_model_sep')

    return model, pad_x, pad_t, learning_rate, batch_size


def load_data_rr(path):
    mat_contents = h.File(path, 'r')

    train_in = mat_contents['train_in'][:]
    train_out = mat_contents['train_out'][:]
    dev_in = mat_contents['dev_in'][:]
    dev_out = mat_contents['dev_out'][:]
    test_in = mat_contents['test_in'][:]
    test_out = mat_contents['test_out'][:]

    sample_freq = mat_contents['sampleFreq'][:]
    phase_step = mat_contents['phaseStep'][:]

    train_in = np.expand_dims(train_in, axis=3)
    dev_in = np.expand_dims(dev_in, axis=3)
    test_in = np.expand_dims(test_in, axis=3)

    train_out = np.expand_dims(train_out, axis=2)
    train_out = np.expand_dims(train_out, axis=3)
    dev_out = np.expand_dims(dev_out, axis=2)
    dev_out = np.expand_dims(dev_out, axis=3)
    test_out = np.expand_dims(test_out, axis=2)
    test_out = np.expand_dims(test_out, axis=3)

    mat_contents.close()

    return train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step


def r2(y_true, y_pred):
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value
