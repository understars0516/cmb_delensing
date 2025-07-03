"""
2D U-Net Model Implementation
============================

This module implements the 2D U-Net network architecture for image-to-image regression tasks,
specifically for CMB delensing applications. U-Net is an encoder-decoder architecture
with skip connections that preserves spatial detail information.

Architecture Features:
- Encoder path: Progressive downsampling to extract features
- Decoder path: Progressive upsampling to recover resolution
- Skip connections: Connect same-level encoder and decoder features
- Batch normalization: Accelerate training and improve stability
- Configurable network depth and filter counts

Author: nisl
Purpose: 2D U-Net implementation for CMB delensing
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate, MaxPool2D, Activation

class unet2D(): 
    """
    2D U-Net Network Builder Class
    
    This class provides complete functionality for building 2D U-Net models with multiple configuration options:
    - Adjustable network depth and filter counts
    - Batch normalization and dropout regularization
    - Multiple activation function choices
    - Flexible convolution layer configuration
    """
    
    def __init__(self, n_filters=16, conv_width=1, 
                 network_depth=4,
                 n_channels=32, x_dim=32, dropout=0.0, 
                 growth_factor=2, batchnorm=True, 
                 momentum=0.9, epsilon=0.001,
                 activation='relu',
                 maxpool=True,
                 psf=False
                 ):
        """
        Initialize U-Net model parameters
        
        Parameters:
        - n_filters: Number of filters in first layer, subsequent layers increase by growth_factor
        - conv_width: Number of convolution layers in each conv block
        - network_depth: Network depth, determines number of downsampling/upsampling layers
        - n_channels: Number of input image channels (1 for grayscale, 3 for RGB)
        - x_dim: Input image dimensions (assumed square)
        - dropout: Dropout regularization rate (0.0-1.0)
        - growth_factor: Filter count growth factor between layers
        - batchnorm: Whether to use batch normalization
        - momentum: Batch normalization momentum parameter
        - epsilon: Batch normalization epsilon parameter
        - activation: Activation function type ('relu', 'elu', 'leaky_relu', etc.)
        - maxpool: Whether to use max pooling for downsampling
        - psf: Point spread function related parameter (currently unused)
        """
        
        # Store all configuration parameters
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.conv_width = conv_width
        self.network_depth = network_depth
        self.x_dim = x_dim
        self.dropout = dropout
        self.growth_factor = growth_factor
        self.batchnorm = batchnorm
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.maxpool = maxpool
        
    def conv_block(self, input_tensor, n_filters, n_layers=1, strides=1, kernel_size=3, 
                   momentum=0.9, maxpool=False, batchnorm=True, layer_num=None):
        """
        Build convolution block
        
        Convolution blocks are the basic building units of U-Net, containing:
        1. Convolution layer
        2. Batch normalization layer (optional)
        3. Activation function
        
        Parameters:
        - input_tensor: Input tensor
        - n_filters: Number of convolution filters
        - n_layers: Number of convolution layers (usually 1 or 2)
        - strides: Convolution stride, >1 for downsampling
        - kernel_size: Convolution kernel size
        - momentum: Batch normalization momentum
        - maxpool: Whether to use max pooling (currently not implemented)
        - batchnorm: Whether to use batch normalization
        - layer_num: Layer number for naming
        
        Returns:
        - Processed tensor
        """
        # Set special naming for downsampling layers
        if layer_num is not None:
            if strides > 1:
                name = 'downsample_{}'.format(layer_num)
        else:
            name = None
        
        x = input_tensor       
        
        # Apply multiple convolution layers
        for _ in range(n_layers):        
            identity = x  # Save input for potential residual connection
            
            # Convolution layer: Extract features
            x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                      padding='same', strides=strides, name=name)(x)

            # Batch normalization: Normalize output, accelerate training
            if batchnorm:
                x = BatchNormalization(momentum=momentum)(x)   
            
            # Activation function: Introduce non-linearity
            x = Activation(self.activation)(x)
            
        # Commented residual connection code
        # if l > 0:
        #     x = Add()([x, identity])
        # x = Activation(self.activation)(x)    
        
        return x           
                     
    
    def build_model(self):
        """
        Build complete U-Net model
        
        U-Net architecture contains three main parts:
        1. Encoder path (contracting path): Progressive downsampling to extract high-level features
        2. Bottleneck layer: Deepest network layer with smallest feature dimensions and richest semantic information
        3. Decoder path (expanding path): Progressive upsampling to recover spatial resolution
        
        Skip connections: Directly connect encoder features at each level to corresponding decoder layer,
        preserving spatial detail information - this is U-Net's core innovation.
        
        Returns:
        - Compiled Keras model
        """
        # Get network configuration parameters
        network_depth = self.network_depth
        n_filters = self.n_filters
        growth_factor = self.growth_factor
        momentum = self.momentum

        # Define input layer
        inputs = keras.layers.Input(shape=(self.x_dim, self.x_dim, self.n_channels), 
                                   name="image_input")
        x = inputs
        concat_down = []  # Store encoder layer outputs for skip connections
        
        # ==================== Encoder Path (Downsampling) ====================
        # Progressive downsampling to extract increasingly high-level features
        for h in range(network_depth):
            # Convolution block: Extract features at current resolution
            x = self.conv_block(x, n_filters, n_layers=self.conv_width, strides=1) 
            concat_down.append(x)  # Save current layer output for skip connections
            
            # Increase filter count for next layer
            n_filters *= growth_factor
            
            # Downsampling: Reduce spatial resolution, expand receptive field
            x = self.conv_block(x, n_filters, n_layers=1, batchnorm=True, strides=2, 
                                maxpool=self.maxpool, layer_num=h+1)
        
        # Reverse skip connection list for decoder use
        concat_down = concat_down[::-1]  
        
        # ==================== Bottleneck Layer ====================
        # Deepest network layer with smallest feature dimensions and richest semantic information
        x = self.conv_block(x, n_filters, n_layers=self.conv_width, strides=1)
        
        # ==================== Decoder Path (Upsampling) ====================
        # Progressive upsampling to recover spatial resolution
        n_filters //= growth_factor  # Start reducing filter count
        
        for h in range(network_depth):
            n_filters //= growth_factor  # Continue reducing filter count
            
            # Transpose convolution: Upsampling to recover spatial resolution
            x = Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
            x = BatchNormalization(momentum=momentum, epsilon=self.epsilon)(x)
            x = Activation(self.activation)(x)
            
            # Skip connection: Connect encoder same-level features with current decoder features
            # This is U-Net's core innovation, preserving spatial detail information
            x = concatenate([x, concat_down[h]])
            
            # Convolution block: Fuse skip connection features
            x = self.conv_block(x, n_filters, n_layers=self.conv_width, kernel_size=3, 
                                strides=1, momentum=self.momentum)   
            
        # ==================== Output Layer ====================
        # Final transpose convolution layer to output final results
        # Output channels equal input channels for pixel-level regression
        last_layer = Conv2DTranspose(self.n_channels, 1, padding="same", 
                                    name="last_layer")(x)
        
        # Create complete model
        model = keras.models.Model(inputs=inputs, outputs=last_layer)

        return model
