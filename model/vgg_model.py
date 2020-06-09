import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.ops import gen_array_ops, nn
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.framework import ops
from tensorflow.keras import layers as L
from tensorflow.keras import Model

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

weights_path = tf.keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH,
                cache_subdir='models')

class VGG16(Model):
    """
    VGG Model with imagenet weights
    """  
    def __init__(self):
        super(VGG16, self).__init__()

        # Block 1
        self.conv_1_1 = L.Conv2D(64, (3, 3), 
                                 padding='same', 
                                 activation='relu', 
                                 name = 'block1_conv1')
        self.conv_1_2 = L.Conv2D(64, (3, 3), 
                                 padding='same', 
                                 activation='relu', 
                                 name = 'block1_conv2')
        self.maxpool_1 = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name = 'block1_pool')
        self.dropout_1 = L.Dropout(0.5)

        # Block 2
        self.conv_2_1 = L.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')
        self.conv_2_2 = L.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')

        self.maxpool_2 = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')
        self.dropout_2 = L.Dropout(0.5)

        # Block 3
        self.conv_3_1 = L.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1')

        self.conv_3_2 = L.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2')

        self.conv_3_3 = L.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3')

        self.maxpool_3 = L.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
        self.dropout_3 = L.Dropout(0.5)

        # Block 4
        self.conv_4_1 = L.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1')

        self.conv_4_2 = L.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')

        self.conv_4_3 = L.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')

        self.maxpool_4 = L.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
        self.dropout_4 = L.Dropout(0.5)

        # Block 5
        self.conv_5_1 = L.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1')

        self.conv_5_2 = L.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2')

        self.conv_5_3 = L.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')

        self.maxpool_5 = L.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flatten = L.Flatten(name='flatten')

    def call(self, x, training = True):
        """
        Override the call method
        """    
        # Block 1
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.maxpool_1(x)
        x = self.dropout_1(x, training)

        # Block 2
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.maxpool_2(x)
        x = self.dropout_2(x, training)

        # Block 3
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.maxpool_3(x)
        x = self.dropout_3(x, training)

        # Block 4
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.maxpool_4(x)
        x = self.dropout_4(x, training)

        # Block 5
        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.maxpool_5(x)

        # Dense
        x = self.flatten(x)

        return x

    def get_features(self, x):

        """
        Get the y_c (final layer before softmax) 
        and the activation map of the last conv layer
        """

        # Block 1
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.maxpool_1(x)

        # Block 2
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.maxpool_2(x)

        # Block 3
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.maxpool_3(x)

        # Block 4
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.maxpool_4(x)

        # Block 5
        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        cam_layer = self.conv_5_3(x)
        x = self.maxpool_5(cam_layer)

        # Dense
        x = self.flatten(x)

        return cam_layer, x

class VGGAwa2(Model):
    def __init__(self, num_classes=10):
        super(VGGAwa2, self).__init__()
        self.base_model = VGG16()
        self.base_model.build((None,224,224,3))
        self.base_model.load_weights(weights_path)
        self.pre_final_dense_1 = L.Dense(1024, name = 'pre_final_1', activation = 'relu')
        self.dropout_1 = L.Dropout(0.5)
        if num_classes < 512:
            self.pre_final_dense_2 = L.Dense(512, name = 'pre_final_2', activation = 'relu')
        else:
            self.pre_final_dense_2 = L.Dense(1024, name = 'pre_final_2', activation = 'relu')
        self.dropout_2 = L.Dropout(0.5)
        self.final_dense = L.Dense(num_classes, name = '10_classes')
        self.softmax =  L.Activation('softmax')

    def call(self, inputs, training = True):
        x = self.base_model(inputs)
        x = self.pre_final_dense_1(x)
        x = self.dropout_1(x, training)
        x = self.pre_final_dense_2(x)
        x = self.dropout_2(x, training)
        x = self.final_dense(x)
        x = self.softmax(x)
        return x

    def get_features(self, inputs):
        features, x = self.base_model.get_features(inputs)
        pre_final_dense = self.pre_final_dense_1(x)
        x = self.pre_final_dense_2(pre_final_dense)
        x = self.final_dense(x)
        x = self.softmax(x)
        return features, pre_final_dense, x
    
    def get_softmax(self, x):
        x = self.pre_final_dense_2(x)
        x = self.final_dense(x)
        x = self.softmax(x)
        return x

    