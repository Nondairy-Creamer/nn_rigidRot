import modelKeras as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import scipy.io as sio
import tensorflow as tf

# define the input path
path = 'C:\\CDocuments\\python\\nn_RigidRot\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_20scenes_2s_10traces_355phi_100Hz_005devFrac.mat'

# load in data set
x_train, y_train, x_dev, y_dev, x_test, y_test = md.load_data_rr(path)

# plot example from training set

plt.figure
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imshow(x_train[9+i, :, :, 0])
    plt.axis('off')
    img.set_cmap('RdBu_r')

plt.show()


# intiialize model
m, size_t, size_x, n_c = x_train.shape
num_filt = 16
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model(input_shape=(size_t, size_x, n_c), filter_shape=(21, 5), num_filter=1)
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_deep(input_shape=(size_t, size_x, n_c), filter_shape=((21, 5), (21, 5)), num_filter=(16, 4))
model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model(input_shape=(size_t, size_x, n_c), filter_shape=(21, 10), num_hrc=1)

# repeat y data to fit output conv size
# y_train = np.tile(y_train, (1, 1, size_x, 1))
# y_dev = np.tile(y_dev, (1, 1, size_x, 1))
# y_test = np.tile(y_test, (1, 1, size_x, 1))

# get rid of edges of y data
# y_train = y_train[:, 0:-1-2*pad_t+1, pad_x:-1-pad_x+1, :]
# y_dev = y_dev[:, 0:-1-2*pad_t+1, pad_x:-1-pad_x+1, :]
# y_test = y_test[:, 0:-1-2*pad_t+1, pad_x:-1-pad_x+1, :]

y_train = y_train[:, 0:-1-2*pad_t+1, :, :]
y_dev = y_dev[:, 0:-1-2*pad_t+1, :, :]
y_test = y_test[:, 0:-1-2*pad_t+1, :, :]

# normalize images
x_train = x_train/np.std(x_train, axis=(1, 2), keepdims=True)
x_dev = x_dev/np.std(x_dev, axis=(1, 2), keepdims=True)
x_test = x_test/np.std(x_test, axis=(1, 2), keepdims=True)

y_train = y_train/np.std(y_train, axis=(1, 2), keepdims=True)
y_dev = y_dev/np.std(y_dev, axis=(1, 2), keepdims=True)
y_test = y_test/np.std(y_test, axis=(1, 2), keepdims=True)

# set up the model and fit it
t = time.time()
adamOpt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
hist = model.fit(x_train, y_train, verbose=2, epochs=2, batch_size=batch_size, validation_data=(x_dev, y_dev))
elapsed = time.time() - t

# grab the loss and R2 over time
#model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
loss = hist.history['loss']
val_loss = hist.history['val_loss']
r2 = hist.history['r2']
val_r2 = hist.history['val_r2']

print('model took ' + str(elapsed) + 's to train')

imageDict = {}
ww = 0

for l in model.layers:
    all_weights = l.get_weights()

    ww += 1

    if len(all_weights) == 2:
        weights = all_weights[0]
        biases = all_weights[1]

        imageDict["weight" + str(ww)] = weights
        imageDict["biases" + str(ww)] = biases

        maxAbsW1 = np.max(np.abs(weights))

        for c in range(weights.shape[2]):
            plt.figure()
            for w in range(weights.shape[3]):
                plt.subplot(2, int(np.ceil(weights.shape[3]/2)), w+1)
                img = plt.imshow(weights[:, :, c, w])
                plt.clim(-maxAbsW1, maxAbsW1)
                plt.axis('off')
                img.set_cmap('RdBu_r')
                # plt.colorbar()

        plt.figure()
        x = np.arange(len(biases))
        plt.scatter(x, biases)

# plot loss and R2
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val loss'])

plt.subplot(1, 2, 2)
plt.plot(r2)
plt.plot(val_r2)
plt.legend(['r2', 'val r2'])

plt.show()

sio.savemat('weights', imageDict)
# plot_model(model, to_file='kerasModel_structure.png')
