import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate, MaxPool2D, Activation

class unet2D(): 
    def __init__(self,n_filters = 16, conv_width=1, 
                 network_depth = 4,
                 n_channels=32, x_dim=32, dropout = 0.0, 
                 growth_factor=2, batchnorm = True, 
                 momentum=0.9, epsilon=0.001,
                 activation='relu',
                 maxpool=True,
                 psf = False
                 ):
        
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
        
        # define all layers
        
    def conv_block(self, input_tensor, n_filters, n_layers=1, strides=1, kernel_size=3, \
                           momentum=0.9, maxpool=False, batchnorm=True, layer_num=None):
        if layer_num is not None:
            if strides > 1:
                name = 'downsample_{}'.format(layer_num)
        else:
            name = None
        
        x = input_tensor       
        
        for _ in range(n_layers):        
            identity = x
            x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                      padding = 'same', strides=strides, name=name)(x)

            if batchnorm:
                x = BatchNormalization(momentum=momentum)(x)   
            x = Activation(self.activation)(x)
        #    if l > 0:
        #        x = Add()([x, identity])
        #    x = Activation(self.activation)(x)    
        return x           
                     
    
    def build_model(self):
        """
        Function to build network with specified architecture parameters
        """
        network_depth = self.network_depth
        n_filters = self.n_filters
        growth_factor = self.growth_factor
        momentum = self.momentum

        ## Start with inputs
        inputs = keras.layers.Input(shape=(self.x_dim, self.x_dim, self.n_channels),name="image_input")
        x = inputs
        concat_down = []
        
        for h in range(network_depth):
            x = self.conv_block(x, n_filters, n_layers=self.conv_width,strides=1) 
            concat_down.append(x)
            n_filters *= growth_factor
            x = self.conv_block(x, n_filters, n_layers=1, batchnorm=True, strides=2, 
                                    maxpool=self.maxpool, layer_num=h+1)
        
        concat_down = concat_down[::-1]  
        x = self.conv_block(x, n_filters, n_layers=self.conv_width, strides=1)
        
        n_filters //= growth_factor
        for h in range(network_depth):
            n_filters //= growth_factor
            x = Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
            x = BatchNormalization(momentum=momentum, epsilon=self.epsilon)(x)
            x = Activation(self.activation)(x)
            x = concatenate([x, concat_down[h]])
            x = self.conv_block(x, n_filters, n_layers=self.conv_width, kernel_size=3, 
                                        strides=1, momentum=self.momentum)   
            
        last_layer = Conv2DTranspose(self.n_channels,1,padding="same",name="last_layer")(x)
        model = keras.models.Model(inputs=inputs,outputs=last_layer)

        return model
