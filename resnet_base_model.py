from __future__ import absolute_import, division, print_function, unicode_literals

# Import Necessary Libraries
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.ops import gen_array_ops, nn
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.framework import ops
from tensorflow.keras import layers as L
from tensorflow.keras import Model
import tensorflow_addons as tfa
from model.resnet50 import ResNet50
from utils.utils import create_dataset, load_dataset
import os

# IMPORT MODEL
model = ResNet50()

# create_dataset()


train_images, train_labels, test_images, test_labels = load_dataset()
print(len(train_images))


# BUILD A TF DATASET
train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_set = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

def read_images(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image, label
    
train_set = train_set.map(read_images)
test_set = test_set.map(read_images)

batch_size = 32
train_set = train_set.shuffle(len(train_labels)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_set = test_set.shuffle(len(test_labels)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# DEFINE LOSSES
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tfa.optimizers.AdamW(5*1e-5, 5*1e-5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(image, labels):
    with tf.GradientTape() as tape:
        prediction = model(image, True)
        loss = cross_entropy(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, prediction)


@tf.function
def test_step(image, labels):
    with tf.GradientTape() as tape:
        prediction = model(image, False)
        loss = cross_entropy(labels, prediction)

    test_loss(loss)
    test_accuracy(labels, prediction)


EPOCHS = 1
history_train_loss = []
history_train_acc = []

history_test_loss = []
history_test_acc = []

for epoch in range(EPOCHS):
    for image, label in train_set:
        train_step(image, label)
    
    for image, label in test_set:
        test_step(image, label)

    template = 'Epoch {}, Train Loss: {}, Train Acc = {}, Val Loss = {}, Val Acc = {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result(),
                        test_loss.result(),
                        test_accuracy.result()))
    
    history_train_loss.append(train_loss.result())
    history_train_acc.append(train_accuracy.result())
    history_test_loss.append(test_loss.result())
    history_test_acc.append(test_accuracy.result())

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

DIRECTORY = 'weights/resnet50'
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

model.save_weights('%s/checkpoint'%DIRECTORY)
model.load_weights('weights/resnet50/checkpoint')

def save_list(history, filename):
    his = np.array(history)
    np.savetxt(filename, his, delimiter = ',')

save_list(history_train_loss, 'logs/%s_train_loss.csv'%('resnet50'))
save_list(history_train_acc, 'logs/%s_history_train_acc.csv'%('resnet50'))
save_list(history_test_loss, 'logs/%s_history_test_loss.csv'%('resnet50'))
save_list(history_test_acc, 'logs/%s_history_test_acc.csv'%('resnet50'))
