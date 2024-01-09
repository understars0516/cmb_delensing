from model import unet2d
from model import unet2p
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)



def ssim_loss(y_true, y_pred):
    y_true_float = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred_float = tf.image.convert_image_dtype(y_pred, tf.float32)

    ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
    loss = 1 - tf.reduce_mean(ssim_val)

    return loss



def rmse(y_true, y_pred):
    y_true_float = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred_float = tf.image.convert_image_dtype(y_pred, tf.float32)
    loss = tf.sqrt(tf.reduce_mean(tf.square(y_true_float - y_pred_float)) + 0.1*0.1) - 0.1
    return loss




TQU = "T"
path = "/home/nisl/Data/CMB_DeLensing/train_arr_1024/"
train_ = np.load(path + "%slens.npy"%TQU)
label_ = np.load(path + "%sunls.npy"%TQU)

train = train_[0:(192*28)]
label = label_[0:(192*28)]


train_test = train_[192*28:192*29]
label_test = label_[192*28:192*29]

train_pred = train_[192*29:192*30]
label_pred = label_[192*29:192*30]


epochs = 10000; batch_size=256; img_rows = img_cols = 512


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = Nest_Net(img_rows, img_cols, 1, deep_supervision=True)
    net.compile(optimizer=tf.optimizers.Adam(learning_rate = 1e-3, beta_1=0.9, beta_2=0.999), loss='mse', metrics=['mae', 'mse'])

history = net.fit(train, label, epochs=epochs, batch_size=batch_size, validation_data=(train_test, label_test))

pred_test = net.predict(train_test)


rearr = hp.read_map("data/rearr_nside2048.fits")


np.save("result/%s_train_loss_epoch%d.npy"%(TQU, epochs), history.history['loss'])
np.save("result/%s_val_loss_epoch%d.npy"%(TQU, epochs), history.history['val_loss'])

np.save("result/%s_epoch-%d_train.npy"%(TQU, epochs), train_test.reshape(-1)[rearr])
np.save("result/%s_epoch-%d_label.npy"%(TQU, epochs), label_test.reshape(-1)[rearr])
np.save("result/%s_epoch-%d_pred.npy"%(TQU, epochs), pred_test.reshape(-1)[rearr])
