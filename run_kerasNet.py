import modelKeras as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import scipy.io as sio
import tensorflow as tf
import datetime


# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_RigidRot'

data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf500_tt1_nt2_hl0-2_vs100_df0-05.mat'
# data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf1000_tt1_nt2_hl0-2_vs100_df0-05.mat'
# data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-05.mat'
# data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt2_nt2_hl0-2_vs100_df0-05.mat'
# data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt4_nt2_hl0-2_vs100_df0-05.mat'

path = data_set_folder + '\\natural_images\\xt\\' + data_set_name

# load in data set
train_in_full, train_out_full, dev_in_full, dev_out_full, test_in_full, test_out_full, sample_freq, phase_step = md.load_data_rr(path)

# parameters of filters
num_runs = 8
filter_time = 0.05  # s
filter_space_list = [15]  # degrees
sum_over_space_list = [False]
num_filt_list = [2]
batch_size_list = [np.power(2, 6)]
learning_rate_list = [0.1]
epoch_list = [500]

# save in a folder with the date
date_str = str(datetime.datetime.now())
date_str = '_'.join(date_str.split(' '))
date_str = '-'.join(date_str.split(':'))
save_folder = data_set_folder + '\\saved_parameters\\'

param_array = np.empty((num_runs, len(sum_over_space_list), len(num_filt_list), len(filter_space_list),
                       len(batch_size_list), len(learning_rate_list), len(epoch_list)), dtype=object)


for run_number_index, run_number in enumerate(range(num_runs)):

    run_begin = time.time()


    for sum_over_space_index, sum_over_space in enumerate(sum_over_space_list):
        for num_filt_index, num_filt in enumerate(num_filt_list):
            for filter_space_index, filter_space in enumerate(filter_space_list):
                for batch_size_index, batch_size in enumerate(batch_size_list):
                    for learning_rate_index, learning_rate in enumerate(learning_rate_list):
                        for epochs_index, epochs in enumerate(epoch_list):
                            filter_indicies_t = int(np.ceil(filter_time*sample_freq))
                            filter_indicies_x = int(np.ceil(filter_space/phase_step))

                            # intiialize model
                            m, size_t, size_x, n_c = train_in_full.shape

                            # model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_hrc=num_filt, sum_over_space=sum_over_space)
                            # model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model_sep(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_hrc=num_filt, sum_over_space=sum_over_space)
                            # model, pad_x, pad_t, _, _ = md.ln_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=num_filt, sum_over_space=sum_over_space)
                            # model, pad_x, pad_t, _, _ = md.ln_model_medulla(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=num_filt, sum_over_space=sum_over_space)
                            model, pad_x, pad_t, _, _ = md.ln_model_flip(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=num_filt, sum_over_space=sum_over_space)
                            # model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_deep_2(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=num_filt, sum_over_space=sum_over_space)
                            # model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_deep(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=(4, 4), sum_over_space=sum_over_space)

                            # format y data to fit with output
                            if sum_over_space:
                                train_out = train_out_full[:, 0:-1 - pad_t + 1, :, :]
                                dev_out = dev_out_full[:, 0:-1 - pad_t + 1, :, :]
                                test_out = test_out_full[:, 0:-1 - pad_t + 1, :, :]
                            else:
                                # repeat y data to fit output conv size
                                train_out_tiled = np.tile(train_out_full, (1, 1, size_x, 1))
                                dev_out_tiled = np.tile(dev_out_full, (1, 1, size_x, 1))
                                test_out_tiled = np.tile(test_out_full, (1, 1, size_x, 1))

                                low_x = int(np.floor(pad_x / 2))
                                high_x = int(np.ceil(pad_x / 2))

                                train_out = train_out_tiled[:, 0:-1-pad_t+1, low_x:-1-high_x+1, :]
                                dev_out = dev_out_tiled[:, 0:-1-pad_t+1, low_x:-1-high_x+1, :]
                                test_out = test_out_tiled[:, 0:-1-pad_t+1, low_x:-1-high_x+1, :]

                            # normalize images
                            train_in = train_in_full/np.std(train_in_full, axis=(1, 2), keepdims=True)
                            dev_in = dev_in_full/np.std(dev_in_full, axis=(1, 2), keepdims=True)
                            test_in = test_in_full/np.std(test_in_full, axis=(1, 2), keepdims=True)

                            # set up the model and fit it
                            t = time.time()
                            adamOpt = optimizers.Adam(lr=learning_rate, decay=0)
                            model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
                            hist = model.fit(train_in, train_out, verbose=0, epochs=epochs, batch_size=batch_size, validation_data=(dev_in, dev_out))
                            elapsed = time.time() - t

                            # grab the loss and R2 over time
                            #model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
                            loss = hist.history['loss']
                            val_loss = hist.history['val_loss']
                            r2 = hist.history['r2']
                            val_r2 = hist.history['val_r2']

                            print('model took ' + str(elapsed) + 's to train')

                            weight_dict = {}
                            ww = 0

                            for l in model.layers:
                                all_weights = l.get_weights()

                                ww += 1

                                if len(all_weights) > 0:
                                    weights = all_weights[0]
                                    if len(all_weights) == 2:
                                        biases = all_weights[1]
                                    else:
                                        biases = [0]

                                    weight_dict['weight' + str(ww)] = weights
                                    weight_dict['biases' + str(ww)] = biases

                            param_dict = {'model_name': model.name,
                                          'R2': r2,
                                          'val_R2': val_r2,
                                          'time': elapsed,
                                          'num_filt': num_filt,
                                          'sum_over_space': sum_over_space,
                                          'filter_time': filter_time,
                                          'filter_space': filter_space,
                                          'sample_freq': int(sample_freq),
                                          'phase_step': int(phase_step),
                                          'stimulus_path': path}

                            param_array[run_number_index, sum_over_space_index, num_filt_index, filter_space_index,
                                        batch_size_index, learning_rate_index, epochs_index] = {'param_dict': param_dict.copy()}

                            param_array[run_number_index, sum_over_space_index, num_filt_index, filter_space_index,
                                        batch_size_index, learning_rate_index, epochs_index]['weight_dict'] = weight_dict.copy()
                            # param_array[run_number_index, sum_over_space_index, num_filt_index, filter_space_index,
                            #            batch_size_index, learning_rate_index, epochs_index]['model'] = model

    run_end = time.time() - run_begin
    print('run took ' + str(run_end/60) + ' minutes to train')
    print('aprox ' + str((num_runs-run_number)*run_end/60) + ' minutes remaining')

output_dict = {'param_array': param_array}
sio.savemat(save_folder + date_str, output_dict)
