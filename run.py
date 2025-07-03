import numpy as np
import tensorflow as tf
from unet import unet_2d
from tensorflow.image import ssim as ssim_cal
from tensorflow.image import psnr as psnr_cal
from tensorflow.keras import backend as K
import os, sys
import math

epochs = int(sys.argv[1]) #2000 
field = str(sys.argv[2]) # "B"
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'

# 创建保存目录
checkpoint_dir = "./checkpoints/check_%s"%field
os.makedirs(checkpoint_dir, exist_ok=True)

boundaries = [1000, 3000]
values = [1e-2, 1e-3, 1e-4]
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
lr = 1e-4

img_w = img_h = 512
# theat_batch in [0, 60, 120, 180, 240]
# phi_batch in [-90, -60, -30, 0, 30, 60]
theta_phi = [[60, -30], [60, -60], [120, 30], [120, 60], [240, -30], [240, -60], [240, 30], [240, 60]]
theta_batch, phi_batch = theta_phi[3]

rot_arr = np.load("rots_npys/rot_arr_%d_%d.npy"%(theta_batch, phi_batch))
rerot_arr = np.load("rots_npys/rerot_arr_%d_%d.npy"%(theta_batch, phi_batch))

rearr = np.load("/home/nisl/Data/CMB_DeLensing/rearr_nside/rearr_nside2048.npy").astype("int")
arr = np.load("/home/nisl/Data/CMB_DeLensing/rearr_nside/arr_nside2048_192x512x512.npy").astype("int")

x_raw = np.load("./train_dataset/%slen_5maps.npy"%field).reshape(-1, img_w, img_h, 1).astype('float32')
y_raw = np.load("./train_dataset/%sunl_5maps.npy"%field).reshape(-1, img_w, img_h, 1).astype('float32')
x = []; y = []
for i in range(5):
    x.append(x_raw[(i*192):((i+1)*192), :, :, 0].reshape(-1)[rearr][rot_arr][arr])
    y.append(y_raw[(i*192):((i+1)*192), :, :, 0].reshape(-1)[rearr][rot_arr][arr])
x = np.array(x).reshape(-1, 512, 512, 1)
y = np.array(y).reshape(-1, 512, 512, 1)

scale = int(sys.argv[3]) 
y = scale * y

batch_size = 16; k_size = 3; dropout_rate = 0.2
TRAIN = 4; VAL = 5; TEST = 0

N_TRAIN = int(TRAIN*192); N_VAL  = int(VAL*192); N_TEST = int(TEST*192)

train_input = x[0:N_TRAIN]
valid_input = x[N_TRAIN:N_VAL]
test_input = x[N_TEST:N_VAL]

train_label = y[0:N_TRAIN]
valid_label = y[N_TRAIN:N_VAL]
test_label = y[N_TEST:N_VAL]

print("train_input, valid_input, test_input shape:", train_input.shape, valid_input.shape, test_input.shape)
print("train_label, valid_label, test_label shape:", train_label.shape, valid_label.shape, test_label.shape)

# ========== 检查点配置 ==========
# 检查点文件路径（每100个epoch保存一次）
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_100.ckpt")
# 训练状态文件
status_file = os.path.join(checkpoint_dir, "training_status.npy")

# 检查是否已有检查点
initial_epoch = 0
if os.path.exists(status_file):
    try:
        status = np.load(status_file, allow_pickle=True).item()
        initial_epoch = status['last_epoch']
        print(f"Resuming training from epoch {initial_epoch}")
        # 计算上一个保存点
        last_saved_epoch = (initial_epoch // 100) * 100
        print(f"Last saved checkpoint at epoch {last_saved_epoch}")
    except:
        print("Error loading training status. Starting from scratch.")
        initial_epoch = 0
else:
    print("No checkpoint found. Starting new training.")

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # 创建模型
    net = unet_2d.unet2D(n_filters=256, conv_width=3, network_depth=3, n_channels=1, x_dim=img_h, dropout=0.2, \
                         growth_factor=2, batchnorm=True, momentum=0.9, epsilon=0.001, activation='relu', maxpool=True)
    net = net.build_model()
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999), 
                loss='logcosh', 
                metrics=['mae', 'mse'])
    
    # 如果存在检查点，加载权重
    if os.path.exists(checkpoint_path + ".index"):
        net.load_weights(checkpoint_path)
        print(f"Loaded model weights from {checkpoint_path}")

# 自定义回调：保存训练状态和每100个epoch保存检查点
class PeriodicCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path, status_file):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.status_file = status_file
        self.last_saved_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1  # epoch从0开始，所以+1
        
        # 保存当前训练状态
        current_status = {
            'last_epoch': current_epoch
        }
        np.save(self.status_file, current_status)
        
        # 每100个epoch保存一次检查点
        if current_epoch % 100 == 0:
            self.model.save_weights(self.checkpoint_path, overwrite=True)
            self.last_saved_epoch = current_epoch
            print(f"\nSaved checkpoint at epoch {current_epoch} to {self.checkpoint_path}")

# 回调列表
callbacks = [
    PeriodicCheckpoint(checkpoint_path, status_file)
]

# 训练模型
history = net.fit(
    train_input, train_label,
    epochs=epochs,
    batch_size=batch_size,
    initial_epoch=initial_epoch,
    validation_data=(valid_input, valid_label),
    callbacks=callbacks
)

# 训练完成后保存最终模型
final_model_path = os.path.join(checkpoint_dir, "final_model.h5")
net.save(final_model_path)
print(f"Saved final model to {final_model_path}")

# 获取训练历史
train_loss = history.history['loss']
val_loss = history.history['val_loss']

pred = net.predict(test_input)/scale
test_label = test_label/scale


pred_raw =[]; test_raw = []; label_raw = []
for i in range(5):
    pred_raw.append(pred[(i*192):((i+1)*192), :, :, 0].reshape(-1)[rearr][rerot_arr])
    label_raw.append(test_label[(i*192):((i+1)*192), :, :, 0].reshape(-1)[rearr][rerot_arr])
    test_raw.append(test_input[(i*192):((i+1)*192), :, :, 0].reshape(-1)[rearr][rerot_arr])






np.save('result/%s_pred_epochs_%d_%d_%d.npy'%(field, epochs, theta_batch, phi_batch), pred_raw)
np.save('result/%s_test_epochs_%d_%d_%d.npy'%(field, epochs, theta_batch, phi_batch), label_raw)
np.save('result/%s_label_epochs_%d_%d_%d.npy'%(field, epochs, theta_batch, phi_batch), test_raw)
np.save('result/%s_loss_train_epcohs_%d_%d_%d.npy'%(field, epochs, theta_batch, phi_batch), train_loss)
np.save('result/%s_loss_val_epochs_%d_%d_%d.npy'%(field, epochs, theta_batch, phi_batch), val_loss)

