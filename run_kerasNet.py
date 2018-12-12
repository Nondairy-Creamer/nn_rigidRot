import modelKeras as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

# define the input path
path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_test.mat'
# path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_2s_10traces_355phi_100Hz_005devFrac.mat'
# path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_2s_10traces_355phi_100Hz_01devFrac.mat'

# load in data set
x_train, y_train, x_dev, y_dev, x_test, y_test = md.load_data_rr(path)

# plot example from training set
"""
plt.figure(1)
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imshow(x_train[9+i, :, :, 0])
    plt.axis('off')
    img.set_cmap('RdBu_r')

plt.show()
"""

# intiialize model
m, size_t, size_x, n_c = x_train.shape
num_filt = 16
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model(input_shape=(size_t, size_x, n_c), filter_shape=(21, 5), num_filter=2)
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_deep(input_shape=(size_t, size_x, n_c), filter_shape=((21, 20), (21, 10)), num_filter=(4, 4))
model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model(input_shape=(size_t, size_x, n_c), filter_shape=(21, 10), num_hrc=4)

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

# set up the model and fit it
t = time.time()
adamOpt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
hist = model.fit(x_train, y_train, verbose=2, epochs=20, batch_size=batch_size, validation_data=(x_dev, y_dev))
elapsed = time.time() - t

# grab the loss and R2 over time
#model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
loss = hist.history['loss']
val_loss = hist.history['val_loss']
r2 = hist.history['r2']
val_r2 = hist.history['val_r2']

print('model took ' + str(elapsed) + 's to train')

for l in model.layers:
    all_weights = l.get_weights()

    if len(all_weights) == 2:
        weights = all_weights[0]
        biases = all_weights[1]

        images = weights+biases

        maxAbsW1 = np.max(np.abs(images))

        for c in range(images.shape[2]):
            plt.figure()
            for w in range(images.shape[3]):
                plt.subplot(2, int(np.ceil(images.shape[3]/2)), w+1)
                img = plt.imshow(images[:, :, c, w])
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

# plot_model(model, to_file='kerasModel_structure.png')
