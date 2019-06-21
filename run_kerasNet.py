import modelKeras as md
import numpy as np
import time
from keras import optimizers
import scipy.io as sio
import datetime

# parameters of filters
no_opponency = False
num_runs = 1
num_runs = 20
filter_time = 0.2  # s
filter_time = 0.3
noise_std_list = [0.1]
# noise_std_list = [0.1, 1.0]
filter_space_list = [15]  # degrees
sum_over_space_list = [False]
num_filt_list = [4]
batch_size_list = [np.power(2, 6)]
batch_size_list = [np.power(2, 8)]
learning_rate_list = [0.1]
learning_rate_list = [0.1]
epoch_list = [200]
epoch_list = [25]
normalize_list = [True]
normalize_list = [False, True]

# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_RigidRot'

# data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-05.mat'
# data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-1_vs100_df0-05_no0.mat'
data_set_name = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt16_hl0-2_vs100_df0-05_no0.mat'

path = data_set_folder + '\\natural_images\\xt\\' + data_set_name

# load in data set
train_in_full, train_out_full, dev_in_full, dev_out_full, test_in_full, test_out_full, sample_freq, phase_step = md.load_data_rr(path)

if no_opponency:
    train_out_full[train_out_full < 0] = 0
    dev_out_full[dev_out_full < 0] = 0
    test_out_full[test_out_full < 0] = 0

m, size_t, size_x, n_c = train_in_full.shape

# save in a folder with the date
date_str = str(datetime.datetime.now())
date_str = '_'.join(date_str.split(' '))
date_str = '-'.join(date_str.split(':'))
save_folder = data_set_folder + '\\saved_parameters\\'

param_dict = []

# generate a list of parameters to perform runs on
for run_number in range(num_runs):
    for sum_over_space in sum_over_space_list:
        for num_filt in num_filt_list:
            for filter_space in filter_space_list:
                for batch_size in batch_size_list:
                    for learning_rate in learning_rate_list:
                        for epochs in epoch_list:
                            for noise_std in noise_std_list:
                                for normalize in normalize_list:

                                    param_dict.append({
                                                       'num_filt': num_filt,
                                                       'sum_over_space': sum_over_space,
                                                       'filter_time': filter_time,
                                                       'filter_space': filter_space,
                                                       'sample_freq': int(sample_freq),
                                                       'phase_step': int(phase_step),
                                                       'learning_rate': learning_rate,
                                                       'data_set_path': path,
                                                       'epochs': epochs,
                                                       'batch_size': batch_size,
                                                       'noise_std': noise_std,
                                                       'normalize_std': normalize,
                                                       'no_opponency': no_opponency,
                                                       })

total_runs = len(param_dict)

# fit all the models
for p_ind, p in enumerate(param_dict):
    run_begin = time.time()

    filter_indicies_t = int(np.ceil(p['filter_time']*p['sample_freq']))
    filter_indicies_x = int(np.ceil(p['filter_space']/p['phase_step']))

    # intiialize model
    # model, pad_x, pad_t = md.ln_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=p['num_filt'], sum_over_space=p['sum_over_space'])
    # model, pad_x, pad_t = md.ln_model_flip(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=p['num_filt'], sum_over_space=p['sum_over_space'])
    # model, pad_x, pad_t = md.conductance_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, 1), num_filter=p['num_filt'], sum_over_space=p['sum_over_space'], fit_reversal=False)
    # model, pad_x, pad_t = md.conductance_model_flip(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, 1), num_filter=p['num_filt'], sum_over_space=p['sum_over_space'], fit_reversal=False)
    model, pad_x, pad_t = md.LNLN_flip(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, 1), num_filter=p['num_filt'], sum_over_space=p['sum_over_space'])

    param_dict[p_ind]['model_name'] = model.name

    # perform modifications to the data and answer sets
    train_in = train_in_full
    dev_in = dev_in_full
    test_in = test_in_full

    train_out = train_out_full
    dev_out = dev_out_full
    test_out = test_out_full

    # normalize images
    if p['normalize_std']:
        train_in = train_in / np.std(train_in_full, axis=(1, 2), keepdims=True)
        dev_in = dev_in / np.std(dev_in_full, axis=(1, 2), keepdims=True)
        test_in = test_in / np.std(test_in_full, axis=(1, 2), keepdims=True)

    train_in = train_in + np.random.randn(train_in.shape[0], train_in.shape[1], train_in.shape[2], train_in.shape[3]) * p['noise_std']
    dev_in = dev_in + np.random.randn(dev_in.shape[0], dev_in.shape[1], dev_in.shape[2], train_in.shape[3]) * p['noise_std']
    test_in = test_in + np.random.randn(test_in.shape[0], test_in.shape[1], test_in.shape[2], train_in.shape[3]) * p['noise_std']

    # format y data to fit with output
    if ~p['sum_over_space']:
        # repeat y data to fit output conv size
        train_out = np.tile(train_out, (1, 1, size_x-pad_x, 1))
        dev_out = np.tile(dev_out, (1, 1, size_x-pad_x, 1))
        test_out = np.tile(test_out, (1, 1, size_x-pad_x, 1))

    train_out = train_out[:, pad_t:, :, :]
    dev_out = dev_out[:, pad_t:, :, :]
    test_out = test_out[:, pad_t:, :, :]


    # set up the model and fit it
    adamOpt = optimizers.Adam(lr=p['learning_rate'], decay=0)
    model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
    hist = model.fit(train_in, train_out, verbose=2, epochs=p['epochs'], batch_size=p['batch_size'], validation_data=(dev_in, dev_out))

    # grab the loss and R2 over time
    #model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
    param_dict[p_ind]['loss'] = hist.history['loss']
    param_dict[p_ind]['val_loss'] = hist.history['val_loss']
    param_dict[p_ind]['r2'] = hist.history['r2']
    param_dict[p_ind]['val_r2'] = hist.history['val_r2']

    # extract the model weights
    weights = []
    biases = []

    for l in model.layers:
        all_weights = l.get_weights()

        if len(all_weights) > 0:
            weights.append(all_weights[0])
            if len(all_weights) == 2:
                biases.append(all_weights[1])
            else:
                biases.append([])

    param_dict[p_ind]['weights'] = weights
    param_dict[p_ind]['biases'] = biases

    run_end = time.time() - run_begin
    print('run ' + str(p_ind+1) + '/' + str(total_runs) + ' took ' + str(run_end/60) + ' minutes to train')
    print('aprox ' + str((total_runs-(p_ind+1))*run_end/60) + ' minutes remaining')

sio.savemat(save_folder + date_str, {'param_dict': param_dict})
