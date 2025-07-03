"""
U-Net++ (Nested U-Net) Model Implementation
==========================================

This module implements the U-Net++ network architecture, an improved version of U-Net.
U-Net++ enhances segmentation performance through dense skip connections and deep supervision.

Core innovations of U-Net++:
1. Nested dense skip connections: More connection paths between different levels
2. Deep supervision: Supervised learning at multiple scales
3. Better feature reuse: Better gradient flow through dense connections

Architecture features:
- Nested U-Net structure forming a dense connection pattern
- Each skip path contains a series of convolution operations
- Supports deep supervision training mode
- Can select outputs at different depths during inference

Author: nisl
Purpose: Advanced U-Net implementation for image segmentation and regression tasks
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout

import numpy as np

# Global parameter settings
smooth = 1.            # Smooth factor for Dice coefficient calculation
dropout_rate = 0.5     # Default dropout rate

def mean_iou(y_true, y_pred):
    """
    Mean Intersection over Union (IoU) calculation function
    
    IoU is a commonly used evaluation metric in segmentation tasks, calculating the ratio
    of intersection to union between predicted and ground truth regions.
    This function calculates IoU at multiple thresholds and takes the average.
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Prediction results
    
    Returns:
    - Mean IoU value
    """
    prec = []
    # Calculate IoU at different thresholds from 0.5 to 1.0 with step 0.05
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)  # Binarize prediction results
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    """
    Dice coefficient calculation function
    
    Dice coefficient is another important evaluation metric in segmentation tasks,
    measuring the similarity between two sets. Values range from 0 to 1, 
    with 1 indicating perfect overlap.
    
    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Prediction results
    
    Returns:
    - Dice coefficient value
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)    # Flatten ground truth labels
    y_pred_f = K.flatten(y_pred)    # Flatten prediction results
    intersection = K.sum(y_true_f * y_pred_f)  # Calculate intersection
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    """
    Combined binary cross-entropy and Dice loss function
    
    Combines advantages of both binary cross-entropy loss and Dice loss:
    - Binary cross-entropy: Good at handling class imbalance
    - Dice loss: Directly optimizes segmentation metrics
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Prediction results
    
    Returns:
    - Combined loss value
    """
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def my_loss(y_true, y_pred):
    """
    Custom loss function
    
    This is a weighted mean squared error loss function that assigns different weights
    to different pixel values. The weight is adjusted by (y_true + 10) to make
    certain regions more important in the loss.
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Prediction results
    
    Returns:
    - Weighted mean squared error loss
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(tf.multiply(tf.multiply(1, y_true+10), tf.math.squared_difference(y_true, y_pred)), axis=-1)

########################################
# 2D Standard Convolution Unit
########################################

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    """
    Standard convolution unit
    
    Basic building block of U-Net++, containing two convolution layers with Dropout after each.
    Uses ELU activation function and He normal initialization.
    
    Architecture: Conv2D -> Dropout -> Conv2D -> Dropout
    
    Parameters:
    - input_tensor: Input tensor
    - stage: Stage identifier for layer naming
    - nb_filter: Number of filters
    - kernel_size: Convolution kernel size
    
    Returns:
    - Processed tensor
    """
    act = 'elu'  # Use ELU activation function, better negative value handling compared to ReLU

    # First convolution layer
    x = Conv2D(nb_filter, (kernel_size, kernel_size), 
               activation=act, 
               name='conv'+stage+'_1', 
               kernel_initializer='he_normal',  # He initialization, suitable for ELU activation
               padding='same', 
               kernel_regularizer=l2(1e-4))(input_tensor)  # L2 regularization to prevent overfitting
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    
    # Second convolution layer
    x = Conv2D(nb_filter, (kernel_size, kernel_size), 
               activation=act, 
               name='conv'+stage+'_2', 
               kernel_initializer='he_normal', 
               padding='same', 
               kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    """
    Build U-Net++ (Nested U-Net) model
    
    The core idea of U-Net++ is to add dense skip connections on top of U-Net.
    Traditional U-Net has only one skip connection path, while U-Net++ has multiple nested paths.
    
    Network structure explanation:
    - X^{i,j} represents the output of the j-th node at the i-th level
    - Each node receives inputs from the same position in the previous level and upsampled from the next level
    - Forms a densely connected network structure
    
    Parameters:
    - img_rows: Input image height
    - img_cols: Input image width  
    - color_type: Number of input channels (1 for grayscale, 3 for color)
    - num_class: Number of output classes
    - deep_supervision: Whether to enable deep supervision
    
    Returns:
    - Built Keras model
    """
    
    # Filter count configuration: Increasing layer by layer
    nb_filter = [32, 64, 128, 256, 512]
    act = 'elu'  # Activation function

    # Set batch normalization axis (for different backends)
    global bn_axis
    bn_axis = 3  # For TensorFlow backend
    
    # Define input layer
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')

    # ==================== First Level (Highest Resolution) ====================
    # First convolution block
    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    # ==================== Second Level ====================
    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    # First upsampling connection: from conv2_1 to conv1_2
    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    # ==================== Third Level ====================
    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    # Second node of second level
    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    # Third node of first level: receives input from conv2_2
    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)  # Dense connection
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    # ==================== Fourth Level ====================
    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    # Continue building nested structure
    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)  # Dense connection
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)  # All previous nodes
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    # ==================== Fifth Level (Bottleneck) ====================
    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    # ==================== Continue Building Remaining Nested Structure ====================
    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    # ==================== Multiple Output Layers (Deep Supervision) ====================
    # An important feature of U-Net++ is the ability to produce outputs at different depths
    # This allows for deep supervision training, improving training effectiveness
    
    # Output layers at various depths
    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', 
                             kernel_initializer='he_normal', padding='same', 
                             kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', 
                             kernel_initializer='he_normal', padding='same', 
                             kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', 
                             kernel_initializer='he_normal', padding='same', 
                             kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', 
                             kernel_initializer='he_normal', padding='same', 
                             kernel_regularizer=l2(1e-4))(conv1_5)

    # Return different models based on whether deep supervision is enabled
    if deep_supervision:
        # Deep supervision mode: return outputs from all levels
        model = Model(inputs=img_input, outputs=[nestnet_output_1, nestnet_output_2, 
                                                nestnet_output_3, nestnet_output_4])
    else:
        # Standard mode: return only the deepest level output
        model = Model(inputs=img_input, outputs=[nestnet_output_4])

    return model


if __name__ == '__main__':
    """
    Test code: Create a 96x96 U-Net++ model and display its structure
    """
    net = Nest_Net(96, 96, 1)
    net.summary()

