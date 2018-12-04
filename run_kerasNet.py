import modelKeras as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

# path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_02s_10traces_40phi_100Hz_005devFrac.mat'
path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_02s_10traces_40phi_100Hz_01devFrac.mat'
# path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_02s_10traces_40phi_100Hz_02devFrac.mat'
# path = 'C:\\CDocuments\\python\\NaturalImages\\xtPlot_natImageCombinedFilteredContrast_2s_6traces_355phi_100Hz_01devFrac.mat'

# load in data set
x_train, y_train, x_dev, y_dev, x_test, y_test = md.load_data_rr(path)

# plot example from training set
plt.figure(1)
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imshow(x_train[9+i, :, :, 0])
    plt.axis('off')
    img.set_cmap('RdBu_r')

plt.show()

# intiialize model
m, size_t, size_x, n_c = x_train.shape
num_filt = 3
model, pad, learning_rate = md.ln_model(input_shape=(size_t, size_x, n_c), filter_shape=(21, size_x), num_filt=num_filt)
# model, pad, learning_rate = md.hrc_model(input_shape=(size_t, size_x, n_c), filter_shape=(21, size_x), num_filt=2)

# get rid of edges of y data
#y_train = y_train[:, 0:-1-2*pad+1, :, :]
#y_dev = y_dev[:, 0:-1-2*pad+1, :, :]
#y_test = y_test[:, 0:-1-2*pad+1, :, :]

y_train = y_train[:, 2*pad-1:-1, :, :]
y_dev = y_dev[:,2*pad-1:-1, :, :]
y_test = y_test[:, 2*pad-1:-1, :, :]

# set up the model and fit it
t = time.time()
adamOpt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
hist = model.fit(x_train, y_train, verbose=2, epochs=20, batch_size=np.power(2, 12), validation_data=(x_dev, y_dev))
elapsed = time.time() - t
model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
loss = hist.history['loss']
val_loss = hist.history['val_loss']
r2 = hist.history['r2']
val_r2 = hist.history['val_r2']

print('model took ' + str(elapsed) + 's to train')

# get model predictions
yHat_train = model.predict(x_train)
yHat_dev = model.predict(x_dev)

# measure model accuracy
train_accuracy = 1-np.sum(np.square(yHat_train - y_train)) / \
    np.sum(np.square(y_train-np.mean(y_train)))
dev_accuracy = 1 - np.sum(np.square(yHat_dev - y_dev)) / \
    np.sum(np.square(y_dev - np.mean(y_dev)))

print('train accuracy: ' + str(train_accuracy))
print('dev accuracy: ' + str(dev_accuracy))

# plot weights
W1 = model.layers[2].get_weights()
W1 = W1[0]

maxAbsW1 = np.max(np.abs(W1))

plt.figure(1)
for w in range(W1.shape[3]):
    plt.subplot(2, int(np.ceil(W1.shape[3]/2)), w+1)
    img = plt.imshow(W1[:, :, 0, w])
    plt.clim(-maxAbsW1, maxAbsW1)
    plt.axis('off')
    img.set_cmap('RdBu_r')
    # plt.colorbar()

plt.figure(2)
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
