import numpy as np
import tensorflow as tf
from model import unet2d
from tensorflow.image import ssim as ssim_cal
from tensorflow.image import psnr as psnr_cal
from tensorflow.keras import backend as K
import os, sys


epochs = 10000

field = sys.argv[1]

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def run_norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def rmse(y_true, y_pred):
    y_true_float = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred_float = tf.image.convert_image_dtype(y_pred, tf.float32)
    loss = tf.sqrt(tf.reduce_mean(tf.square(y_true_float - y_pred_float)) + 0.001*0.001) - 0.001
    return loss

boundaries = [750, 2000]
values = [1e-3, 1e-4, 1e-5]
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
lr = 1e-4


TQU = "T"
theta = 0; phi = 0
x = np.load("sim_data/rot_map_data/%slens_rot_theta_%d_phi_%d.npy"%(TQU, theta, phi)).reshape(-1, 512, 512, 1)
y = np.load("sim_data/rot_map_data/%sunls_rot_theta_%d_phi_%d.npy"%(TQU, theta, phi)).reshape(-1, 512, 512, 1)
scale = 1
y = scale * y


img_w = img_h = 512
batch_size = 32; k_size = 3; dropout_rate = 0.2
TRAIN = 4; VAL = 5; TEST = 0

N_TRAIN = int(TRAIN*192); N_VAL  = int(VAL*192); N_TEST = int(TEST*192)

train_input = x[0:N_TRAIN]; #train_input = gray2rgb(train_input).numpy()
valid_input   = x[N_TRAIN:N_VAL]; #valid_input = gray2rgb(valid_input).numpy()
test_input  = x[N_TEST:N_VAL]; #test_input = gray2rgb(test_input).numpy()

train_label = y[0:N_TRAIN]; #train_label = gray2rgb(train_label).numpy()
valid_label   = y[N_TRAIN:N_VAL];# valid_label = gray2rgb(valid_label).numpy()
test_label  = y[N_TEST:N_VAL]; #test_label = gray2rgb(test_label).numpy()

print("train_input, valid_input, test_input shape:", train_input.shape, valid_input.shape, test_input.shape)
print("train_label, valid_label, test_label shape:", train_label.shape, valid_label.shape, test_label.shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = unet2d.unet2D(n_filters=32, conv_width=3, network_depth=3, n_channels=1, x_dim=img_h, dropout=0.2, \
                         growth_factor=2, batchnorm=True, momentum=0.9, epsilon=0.001, activation='relu', maxpool=True)
    net = net.build_model()
    net.compile(optimizer=tf.optimizers.Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999), loss='mse', metrics=['mae', 'mse'])
    #net.compile(optimizer=tf.optimizers.Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999), loss=tf.keras.losses.LogCosh(), metrics=['mae', 'mse'])


history = net.fit(train_input, train_label, epochs=epochs, batch_size=batch_size, validation_data=(valid_input, valid_label))
train_loss = history.history['loss']
val_loss = history.history['val_loss']

pred = net.predict(test_input) / scale
np.save('result/%s_pred_epochs_%d.npy'%(field, epochs), pred)
np.save('result/%s_test_epochs_%d.npy'%(field, epochs), test_input)
np.save('result/%s_label_epochs_%d.npy'%(field, epochs), test_label)
np.save('result/%s_loss_train_epcohs_%d.npy'%(field, epochs), train_loss)
np.save('result/%s_loss_val_epochs_%d.npy'%(field, epochs), val_loss)

