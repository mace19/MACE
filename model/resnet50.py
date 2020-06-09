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

class ResNet50(Model):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.base_model = tf.keras.applications.ResNet50(include_top = False, classes = 10)
        self.base_model.build((None,224,224,3))
        self.flatten = tf.keras.layers.Flatten()
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
        x = self.flatten(x)
        x = self.pre_final_dense_1(x)
        x = self.dropout_1(x, training)
        x = self.pre_final_dense_2(x)
        x = self.dropout_2(x, training)
        x = self.final_dense(x)
        x = self.softmax(x)
        return x

    def get_features(self, inputs):
        features = self.base_model(inputs)
        x = self.flatten(features)
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