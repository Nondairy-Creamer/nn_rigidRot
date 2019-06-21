from keras.layers import Add, multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv2D, subtract, add, Lambda, Conv3D
from keras.models import Model
from keras.initializers import glorot_uniform
#import keras as TFK
import keras.backend as K
import h5py as h
import numpy as np
from keras.layers import *
import scipy.io as sio
from keras import regularizers
from keras.layers import Layer

def l1_reg_sqrt(weight_matrix):
    return 0.1 * K.sum(K.sqrt(K.abs(weight_matrix)))


def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))

    conv1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1',
                   kernel_initializer=glorot_uniform(seed=None),
                   activation='relu',
                   kernel_regularizer=l1_reg_sqrt)(image_in)

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

    return model, pad_x, pad_t


def conductance_model(input_shape=(11, 9, 1), filter_shape=(21, 1), num_filter=2, sum_over_space=True, fit_reversal=False):
    reg_val = 1

    v_leak = 0
    v_exc = 60
    v_inh = -30
    g_leak = 1

    pad_x = int((filter_shape[1] - 1))+2
    pad_t = int((filter_shape[0] - 1))

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(image_in)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(image_in)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(image_in)

    g1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g1',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)(s1)

    g2 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g2',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)(s2)

    g3 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g3',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)(s3)

    if fit_reversal:
        expand_last = Lambda(lambda lam: K.expand_dims(lam, axis=-1))
        squeeze_last = Lambda(lambda lam: K.squeeze(lam, axis=-1))

        g1 = expand_last(g1)
        g2 = expand_last(g2)
        g3 = expand_last(g3)

        numerator_in = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1, g2, g3])
        numerator = Conv3D(1, (1, 1, 1), strides=(1, 1, 1), name='create_numerator',
                           kernel_initializer=glorot_uniform(seed=None),
                           kernel_regularizer=l1_reg_sqrt,
                           use_bias=False)(numerator_in)

        denominator = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1, g2, g3])
        vm = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])
        vm = squeeze_last(vm)

    else:
        g1_v_inh = Lambda(lambda lam: lam * v_inh)(g1)
        g2_v_exc = Lambda(lambda lam: lam * v_exc)(g2)
        g3_v_inh = Lambda(lambda lam: lam * v_inh)(g3)

        numerator = Lambda(lambda inputs: inputs[0] + inputs[1] + inputs[2])([g1_v_inh, g2_v_exc, g3_v_inh])
        denominator = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1, g2, g3])
        vm = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])

    vm_bias = BiasLayer()(vm)
    vm_rect = Lambda(lambda lam: K.relu(lam))(vm_bias)

    if sum_over_space:
        conv_x_size = vm.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(vm_rect)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='conductance_model')

    return model, pad_x, pad_t


def conductance_model_flip(input_shape=(11, 9, 1), filter_shape=(21, 1), num_filter=2, sum_over_space=True, fit_reversal=False):
    reg_val = 1

    v_leak = 0
    v_exc = 60
    v_inh = -30
    g_leak = 1

    assert(np.mod(num_filter, 2) == 0)
    num_filter = int(num_filter/2)

    pad_x = int((filter_shape[1] - 1))+2
    pad_t = int((filter_shape[0] - 1))

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(image_in)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(image_in)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(image_in)

    g1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g1',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)

    g2 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g2',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)

    g3 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g3',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)

    g2_both = g2(s2)

    g1_1 = g1(s1)
    g1_2 = g1(s3)

    g3_1 = g3(s3)
    g3_2 = g3(s1)

    if fit_reversal:
        expand_last = Lambda(lambda lam: K.expand_dims(lam, axis=-1))
        squeeze_last = Lambda(lambda lam: K.squeeze(lam, axis=-1))

        g2_both = expand_last(g2_both)
        g1_1 = expand_last(g1_1)
        g1_2 = expand_last(g1_2)
        g3_1 = expand_last(g3_1)
        g3_2 = expand_last(g3_2)

        numerator_in_1 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_1, g2_both, g3_1])
        numerator_in_2 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_2, g2_both, g3_2])

        numerator_comb = Conv3D(1, (1, 1, 1), strides=(1, 1, 1), name='create_numerator',
                                kernel_initializer=glorot_uniform(seed=None),
                                kernel_regularizer=l1_reg_sqrt,
                                use_bias=False)
        reverseLayer2 = Lambda(lambda x: K.reverse(x, axes=4))

        numerator_1 = numerator_comb(numerator_in_1)
        denominator_1 = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1_1, g2_both, g3_1])
        vm_1 = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator_1, denominator_1])

        numerator_2 = numerator_comb(reverseLayer2(numerator_in_2))
        denominator_2 = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1_2, g2_both, g3_2])
        vm_2 = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator_2, denominator_2])

        vm_1 = squeeze_last(vm_1)
        vm_2 = squeeze_last(vm_2)

    else:
        g2_both_v_exc = Lambda(lambda lam: lam * v_exc)(g2_both)

        g1_1_v_inh = Lambda(lambda lam: lam * v_inh)(g1_1)
        g1_2_v_inh = Lambda(lambda lam: lam * v_inh)(g1_2)

        g3_1_v_inh = Lambda(lambda lam: lam * v_inh)(g3_1)
        g3_2_v_inh = Lambda(lambda lam: lam * v_inh)(g3_2)

        # for the first detector
        numerator_1 = Lambda(lambda inputs: inputs[0] + inputs[1] + inputs[2])([g1_1_v_inh, g2_both_v_exc, g3_1_v_inh])
        denomenator_1 = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1_1, g2_both, g3_1])
        vm_1 = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator_1, denomenator_1])

        # for the second detector
        numerator_2 = Lambda(lambda inputs: inputs[0] + inputs[1] + inputs[2])([g1_2_v_inh, g2_both_v_exc, g3_2_v_inh])
        denomenator_2 = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1_2, g2_both, g3_2])
        vm_2 = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator_2, denomenator_2])

    bias_layer = BiasLayer()
    vm_1_bias = bias_layer(vm_1)
    vm_2_bias = bias_layer(vm_2)

    vm_1_rect = Lambda(lambda lam: K.relu(lam))(vm_1_bias)
    vm_2_rect = Lambda(lambda lam: K.relu(lam))(vm_2_bias)

    vm = subtract([vm_1_rect, vm_2_rect])

    if sum_over_space:
        conv_x_size = vm.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(vm)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='conductance_model_flip')

    return model, pad_x, pad_t


def LNLN_flip(input_shape=(11, 9, 1), filter_shape=(21, 1), num_filter=2, sum_over_space=True):
    assert(np.mod(num_filter, 2) == 0)
    num_filter = int(num_filter/2)

    pad_x = int((filter_shape[1] - 1))+2
    pad_t = int((filter_shape[0] - 1))

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(image_in)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(image_in)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(image_in)

    g1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g1',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)

    g2 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g2',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)

    g3 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g3',
                kernel_initializer=glorot_uniform(seed=None),
                activation='relu',
                kernel_regularizer=l1_reg_sqrt)

    g2_both = g2(s2)

    g1_1 = g1(s1)
    g1_2 = g1(s3)

    g3_1 = g3(s3)
    g3_2 = g3(s1)

    expand_last = Lambda(lambda lam: K.expand_dims(lam, axis=-1))
    squeeze_last = Lambda(lambda lam: K.squeeze(lam, axis=-1))

    g2_both = expand_last(g2_both)
    g1_1 = expand_last(g1_1)
    g1_2 = expand_last(g1_2)
    g3_1 = expand_last(g3_1)
    g3_2 = expand_last(g3_2)

    numerator_in_1 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_1, g2_both, g3_1])
    numerator_in_2 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_2, g2_both, g3_2])

    numerator_comb = Conv3D(1, (1, 1, 1), strides=(1, 1, 1), name='create_numerator',
                            kernel_initializer=glorot_uniform(seed=None),
                            kernel_regularizer=l1_reg_sqrt,
                            use_bias=True,
                            activation='relu')

    reverse_layer = Lambda(lambda x: K.reverse(x, axes=4))

    numerator_1 = numerator_comb(numerator_in_1)
    numerator_2 = numerator_comb(reverse_layer(numerator_in_2))

    numerator_1 = squeeze_last(numerator_1)
    numerator_2 = squeeze_last(numerator_2)

    vm = subtract([numerator_1, numerator_2])

    if sum_over_space:
        conv_x_size = vm.get_shape().as_list()[2]
    else:
        conv_x_size = 1

    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=l1_reg_sqrt,
                             use_bias=False)(vm)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='LNLN_flip')

    return model, pad_x, pad_t


def ln_model_medulla(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
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

    return model, pad_x, pad_t


def ln_model_flip(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
    reg_val = 1

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    assert (np.mod(num_filter, 2) == 0)
    num_filter = int(num_filter / 2)

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

    return model, pad_x, pad_t


def hrc_model(input_shape=(11, 9, 1), filter_shape=(21, 2), num_hrc=1, sum_over_space=True):
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

    return model, pad_x, pad_t


def hrc_model_sep(input_shape=(11, 9, 1), filter_shape=(21, 2), num_hrc=1, sum_over_space=True):
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

    return model, pad_x, pad_t


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
    # r2_value = 1 - K.mean(K.sum(K.square(y_pred - y_true), axis=[1, 2])/K.sum(K.square(y_true - K.mean(y_true)), axis=[1, 2]))
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value


class BiasLayer(Layer):
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zero',
                                    trainable=True)
        super(BiasLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


