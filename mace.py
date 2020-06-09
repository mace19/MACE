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
from model.vgg_model import VGGAwa2
from utils.utils import create_dataset, load_dataset
import os
import copy

class ProtoNet(Model):
    
    def __init__(self):
        super(ProtoNet, self).__init__()
        
        # Flattens a feature map
        self.flatten = L.Flatten()
        
        # The network to convert a feature map to 32 dimensional concept
        self.dense_1 = L.Dense(128, activation = 'relu')
        self.dense_2 = L.Dense(64, activation = 'relu')
        self.dense_3 = L.Dense(32, activation = 'tanh')
        self.l2_normalize = L.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        
    def call(self, inputs):
        
        flattened_input = self.flatten(inputs)
        output = self.dense_1(flattened_input)
        output = self.dense_2(output)
        output = self.dense_3(output)
        norm = tf.reduce_sum(tf.square(output), axis = -1, keepdims = True) # commented newly
        # since norm would always be positive, tanh should be used to find the weight
        weight = norm
        output = self.l2_normalize(output)
        
        return output, weight
        

class NonNegMaxNorm():
  
    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        
        # Make it Non Neg
        w = w * tf.cast(tf.math.greater_equal(w, 0.), tf.float32)
        w = tf.clip_by_norm(w, self.max_value, axes = [self.axis])
        return w

    def get_config(self):
        return {'max_value': self.max_value, 'axis': self.axis}


import math 

class InterpretCNN(Model):
    def __init__(self, num_classes = 10, proto_per_class = 10, first_dense_dim = 1024):
        super(InterpretCNN, self).__init__()

        self.num_classes = num_classes
        self.proto_per_class = proto_per_class
        self.num_prototypes = num_classes*proto_per_class
        
        self.feature_extract = L.Conv2D(self.num_prototypes , (1,1) , activation = 'relu')

        # This maps a feature map into a local prototype
        self.protonet = [ProtoNet() for i in range(self.num_classes)]
        
        # This finds the concept relevance
        self.concept_relevance  = [L.Dense(1) for i in range(self.num_classes)]
        self.softmax = L.Activation('sigmoid')
        
        # Final Dense Layer
        self.final_dense = L.Dense(first_dense_dim, activation = 'relu')
        self.prune = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1,]
                 ,[0, 1, 0, 0, 1, 0, 0, 0, 0, 0,]
                 ,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0,]
                 ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1,]
                 ,[1, 1, 1, 0, 0, 0, 1, 0, 0, 0,]
                 ,[0, 1, 1, 0, 1, 0, 0, 0, 1, 0,]
                 ,[1, 0, 1, 0, 1, 0, 0, 0, 0, 1,]
                 ,[0, 1, 1, 0, 0, 0, 0, 1, 0, 0,]
                 ,[0, 1, 0, 0, 0, 0, 1, 0, 0, 0,]
                 ,[0, 0, 0, 0, 0, 1, 1, 0, 0, 0,]]
        
    def get_relevance_part(self):
        return self.concept_relevance
    
    def call(self, inputs):
        
        batch_size = inputs.shape[0]
        concept_maps = self.feature_extract(inputs)
        concept_maps = tf.unstack(concept_maps, axis = -1)
        
        # List of lists for prototypes
        class_prototypes = [[] for i in range(self.num_classes)]
        class_protoweight = [[] for i in range(self.num_classes)]
        class_protoweighted = [[] for i in range(self.num_classes)]
        
        # Get Concepts
        for i, concept_map in enumerate(concept_maps):
            
            # class index ci
            ci = i//self.proto_per_class
            
            # Pass the featuremap to the protonet
            prototype, weight = self.protonet[ci](concept_map)
            class_prototypes[ci].append(prototype)
            class_protoweight[ci].append(weight)
            if self.prune[ci][i%self.proto_per_class]:
                prototype = 0*prototype
            class_protoweighted[ci].append(prototype)
        
        # relevances and concepts
        relevances, concepts = [], []
        # Get Relevances
        for i, cprototypes in enumerate(class_protoweighted):
            # class prototypes to tensors
            # (batch_size, embedding_size * prototypes per class)
            cps = tf.concat(cprototypes, axis = 1)
            relev = self.concept_relevance[i](cps)
            # Add Relevances and concepts
            relevances.append(relev)
            concepts.append(cps);
        
        # (Batch_size, num_classes)
        relevances = tf.concat(relevances, axis = 1)
        relevances = self.softmax(relevances)
        expanded_relevances = tf.expand_dims(relevances, axis = -1)
        
        # (Batch_size, embedding_size * num prototypes)
        concepts = tf.concat(concepts, axis = 1)
       
        expanded_relevances = tf.broadcast_to(expanded_relevances, [batch_size,self.num_classes,32*self.proto_per_class])
        expanded_relevances = tf.reshape(expanded_relevances, concepts.shape)
        
        # Weight the concepts
        weighted_concepts = expanded_relevances*concepts
        dense_layer = self.final_dense(weighted_concepts)
        
        return class_prototypes, class_protoweight, relevances, dense_layer, class_protoweighted
    
    
    def get_vis_local(self,inputs):
        batch_size = inputs.shape[0]
        concept_maps = self.feature_extract(inputs)
        concept_maps = tf.unstack(concept_maps, axis = -1)
        
        # List of lists for prototypes
        class_prototypes = [[] for i in range(self.num_classes)]
        class_protoweight = [[] for i in range(self.num_classes)]
        class_protoweighted = [[] for i in range(self.num_classes)]
        
        # Get Concepts
        for i, concept_map in enumerate(concept_maps):
            
            # class index ci
            ci = i//self.proto_per_class
            
            # Pass the featuremap to the protonet
            prototype, weight = self.protonet[ci](concept_map)
            class_prototypes[ci].append(prototype)
            class_protoweight[ci].append(weight)
            if self.prune[ci][i%self.proto_per_class]:
                prototype=0*prototype
            class_protoweighted[ci].append(prototype)
        
        # relevances and concepts
        relevances, concepts = [], []
        # Get Relevances
        for i, cprototypes in enumerate(class_protoweighted):
            # class prototypes to tensors
            # (batch_size, embedding_size * prototypes per class)
            cps = tf.concat(cprototypes, axis = 1)
            relev = self.concept_relevance[i](cps)
            # Add Relevances and concepts
            relevances.append(relev)
            concepts.append(cps);
        
        # (Batch_size, num_classes)
        relevances = tf.concat(relevances, axis = 1)
        relevances = self.softmax(relevances)
        
        return concept_maps, class_prototypes, class_protoweight, relevances, class_protoweighted
    
    
class InterpretClassification(Model):
    def __init__(self, num_classes = 10, proto_per_class = 10, first_dense_dim = 1024):
        super(InterpretClassification, self).__init__()
        self.basemodel = VGGAwa2()
        self.basemodel.build((None, 224, 224, 3))
        self.basemodel.load_weights('weights/vgg_base/checkpoint')
        self.basemodel.trainable=False
        self.interpret_layer = InterpretCNN(num_classes, proto_per_class, first_dense_dim)
        
    def call(self, inputs):
        features, first_dense, final_output = self.basemodel.get_features(inputs)
        class_prototypes, class_protoweight, relevances, dense_layer_predict, class_protoweighted = self.interpret_layer(features)
        final_predict = self.basemodel.get_softmax(dense_layer_predict)
        return class_prototypes, class_protoweight, relevances, dense_layer_predict, first_dense, final_predict, final_output, class_protoweighted
    
    def get_concept_maps(self, inputs):
        features, first_dense, final_output = self.basemodel.get_features(inputs)
        return self.interpret_layer.get_concept_maps(features)
    
    def get_vis_local_maps(self,inputs):
        features, _, _ = self.basemodel.get_features(inputs)
        return self.interpret_layer.get_vis_local(features)
    
    def get_vis_global_maps(self):
        return self.interpret_layer.get_vis_global()
    
    def check_global_concepts(self):
        dense_predict = self.interpret_layer.checkGlobal()
        return self.basemodel.get_softmax(dense_predict)
        
    
meta_model = InterpretClassification()


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

def read_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image

def read_images_2(image, label):
    img = tf.map_fn(read_image, image, dtype = tf.float32)
    return img, label
    

batch_size = 32
tlen = len(train_labels)
test_set = test_set.shuffle(len(test_labels)).batch(batch_size).map(read_images_2).prefetch(tf.data.experimental.AUTOTUNE)
train_set = train_set.shuffle(tlen)
train_set = train_set.batch(batch_size).map(read_images_2).prefetch(tf.data.experimental.AUTOTUNE)


@tf.function
def get_anchors(batch_size, num_concepts):
    anchors_indices = tf.random.uniform(shape=(num_concepts,), maxval = batch_size, dtype = tf.dtypes.int32)
    index_mapper = batch_size*(tf.range(num_concepts))
    return anchors_indices + index_mapper

@tf.function
def pairwise_dist_map(anchors, embeddings):
    # RETURNS PAIRWISE DISTANCE FOR EACH CONCEPT
    return tf.map_fn(lambda x: tf.reduce_sum(tf.square(x - embeddings), axis = 1), anchors)

def sample_negative(avg_positive, masked_negative):

    # GIVEN A PANCHOR, IT GETS THE SEMI-HARD NEGATIVE
    def get_negative(panchor, nanchors):
        mask = tf.math.greater(nanchors, panchor)
        applied_map = tf.boolean_mask(nanchors, mask)
        semi_hard_negative = tf.reduce_min(applied_map) 
        return semi_hard_negative if tf.math.is_finite(semi_hard_negative) else tf.constant(0.0)

    # FOR EACH ELEMENT IN POSITIVE MASK, FIND A SEMI-HARD NEGATIVE
    return tf.map_fn(
      lambda x: get_negative(x[0], x[1]), (avg_positive, masked_negative), dtype = tf.float32
    )

@tf.function
def custom_triplet_loss(local_prototypes, margin = 0.2):
    """
    Memory Efficient Triplet Loss
    """
    batch_size, embedding_size, num_concepts = local_prototypes.shape

    # EMBEDDING: (1st CONCEPT FOR BATCH_SIZE, 2ND CONCEPT FOR BATCH_SIZE)
    # SHAPE = (BATCH_SIZE * NUM_CONCEPTS, EMBEDDING_SIZE)
    transposed_prototype = tf.transpose(local_prototypes, [2,0,1])
    embeddings = tf.reshape(transposed_prototype, (batch_size*num_concepts, embedding_size))

    # MASKS = (NUM_CONCEPTS, NUM_CONCEPT*BATCH_SIZE)
    concepts_range = tf.expand_dims(tf.range(num_concepts), -1)
    column_mask = tf.broadcast_to(concepts_range, (num_concepts, batch_size))
    column_mask = tf.reshape(column_mask, (num_concepts*batch_size,1))
    column_mask = tf.transpose(
      tf.broadcast_to(column_mask, (num_concepts*batch_size, num_concepts))
    )

    positive_mask = tf.math.equal(concepts_range, column_mask)
    negative_mask = tf.math.logical_not(positive_mask)

    # DIMENSION OF ANCHORS = (NUM_CONCEPTS, EMBEDDING_SIZE)
    anchor_indices = get_anchors(batch_size, num_concepts)
    anchors = tf.gather(embeddings, anchor_indices)

    # PAIRWISE DISTANCE = (NUM_CONCEPTS, NUM_CONCEPTS*BATCH_SIZE)
    pairwise_distance = pairwise_dist_map(anchors, embeddings)

    # FIND HARDEST POSITIVE ANCHOR
    masked_positive = tf.boolean_mask(pairwise_distance, positive_mask)
    masked_positive = tf.reshape(masked_positive, (num_concepts, batch_size))
    avg_positive = tf.reduce_mean(masked_positive, axis = 1)

    # SEMI HARD NEGATIVE
    masked_negative = tf.boolean_mask(pairwise_distance, negative_mask)
    masked_negative = tf.reshape(masked_negative, (num_concepts, batch_size*(num_concepts - 1)))
    hardest_negative = tf.expand_dims(sample_negative(avg_positive, masked_negative), -1)

    loss = tf.reduce_mean(
    tf.math.maximum(
        masked_positive - hardest_negative + margin, 0)    
    )

    return loss


@tf.function
def new_triplet_loss(prototypes, margin = 0.2):
    """
    Memory Efficient Triplet Loss
    prototypes.shape = (batch_size, number_of_prototypes, embedding_size)
    """
    
    # FIND HARDEST POSITIVE ACROSS BATCH
    p_prototypes = tf.transpose(prototypes, [1,0,2])
    p_prototypes = tf.expand_dims(p_prototypes, -1)
    p_ = p_prototypes
    p_t = tf.transpose(p_prototypes, [0, 3, 2, 1])
    
    pairwise = tf.reduce_sum(tf.square(p_ - p_t), axis = 2)
    # Diagonals are 0. For each row take the one with the max.
    hardest_p = tf.math.reduce_mean(pairwise, axis = - 1) # (number_of_prototypes, batch_size)
    hardest_p = tf.transpose(hardest_p, [1,0]) # (batch_size, number_of_prototypes)
    
    # FIND HARDEST NEGATIVE ACROSS CONCEPTS
    n_prototypes = tf.expand_dims(prototypes, -1)
    n_ = n_prototypes
    n_t = tf.transpose(n_prototypes, [0, 3, 2, 1])
    
    pairwise = tf.reduce_sum(tf.square(n_ - n_t), axis = 2)
    semi_masked = tf.math.greater(pairwise, tf.expand_dims(hardest_p, -1))
    not_semi_masked = tf.math.logical_not(semi_masked)
    # Diagonals are 0. Make diagonals 4 + 1e-6 (largest distance) For each row take the one with the min. 
    eye = (4 + 1e-6)*tf.cast(not_semi_masked, dtype = tf.float32)
    pairwise = eye + pairwise*tf.cast(semi_masked, dtype = tf.float32)
    
    hardest_n = tf.math.reduce_min(pairwise, axis = - 1) # (batch_size, number_of_prototypes)
    
    loss = tf.reduce_mean(
                tf.math.maximum(
                    hardest_p - hardest_n + margin, 0   
                ), axis = -1
        )

    return loss


mse = tf.keras.losses.MeanSquaredError()
kl = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.optimizers.Adam(0.0001)

train_mse_dense = tf.keras.metrics.Mean(name='train_mse_dense')
train_kl = tf.keras.metrics.Mean(name='train_kl')
train_revelance = tf.keras.metrics.Mean(name='train_kl_uniform')
train_loss_cos = tf.keras.metrics.Mean(name='train_coss')
train_loss_triplet = tf.keras.metrics.Mean(name='train_triplet')

train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
rev_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='rev_acc')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        class_prototypes, class_protoweight, relevances, dense_layer_predict, first_dense, final_predict, final_output, class_protoweighted = meta_model(images)
        # MSE with pre dense layer
        mse_loss = mse(first_dense, dense_layer_predict)
        # KL with original distribution
        kl_loss = kl(final_output, final_predict)
        revelance_loss = kl(final_output, relevances)
        
        triplet_loss = 0
        for cp in class_prototypes:
            # (batch_size, embedding_size, num_prototypes)
            prototype = tf.stack(cp, axis = 2)
            triplet_loss += 0.1*new_triplet_loss(tf.transpose(prototype, [0,2,1]), 1)
       
        # TOTAL LOSS
        loss = mse_loss + kl_loss + revelance_loss + triplet_loss

    gradients = tape.gradient(loss, meta_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, meta_model.trainable_variables))
    
    train_mse_dense(mse_loss)
    train_kl(kl_loss)
    train_revelance(revelance_loss)
    train_loss_triplet(triplet_loss)
    train_acc(tf.math.argmax(final_output,axis=1), final_predict)
    

@tf.function
def test_step(images, labels):
    class_prototypes, class_protoweight, relevances, dense_layer_predict, first_dense, final_predict, final_output, class_protoweighted = meta_model(images)
    test_acc(tf.math.argmax(final_output,axis=1), final_predict)
    rev_acc(tf.math.argmax(final_output,axis=1), relevances)
    

train_kl_loss = []
train_mse_dense_loss = []
train_relevance_loss = []
train_loss_triplet_loss = []

EPOCHS = 50
for epoch in range(EPOCHS):
    for image, label in train_set:
        train_step(image, label)
    
    for image, label in test_set:
        test_step(image, label)

    train_kl_loss.append(train_kl.result())
    train_mse_dense_loss.append(train_mse_dense.result())
    train_relevance_loss.append(train_revelance.result())
    train_loss_triplet_loss.append(train_loss_triplet.result())
    template = 'Epoch {}, MSE Loss: {}, KL Loss: {}, Triplet Loss: {}, Relevance Loss: {}, Train Acc: {}, Test Acc:{}, Rev Acc: {}'
    print(template.format(epoch+1,
                        train_mse_dense.result(),
                        train_kl.result(),
                        train_loss_triplet.result(),
                        train_revelance.result(),
                        train_acc.result(),
                        test_acc.result(),
                        rev_acc.result()
                        )
    )

    # Reset the metrics for the next epoch
    train_mse_dense.reset_states()
    train_kl.reset_states()
    train_loss_triplet.reset_states()
    train_revelance.reset_states()
    train_acc.reset_states()
    test_acc.reset_states()
    rev_acc.result()


DIRECTORY = 'weights'
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

meta_model.save_weights('%s/checkpoint'%DIRECTORY)
meta_model.load_weights('weights/checkpoint')

"""
Code for visualizing the concepts

"""

DIRECTORY = 'weights'
dir_v = 'visualizations/'
if not os.path.exists(dir_v):
    os.makedirs(dir_v)


def load_dataset(dataset_dir = 'datasets/awa2'):
    train_images, train_labels = [], []

    with open('%s/train.txt' % dataset_dir) as fp:
        for entry in fp.readlines():
            attrib = entry.strip().split(':')
            train_labels.append(int(attrib[0]))
            train_images.append(attrib[1])

# BUILD A TF DATASET
train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

def read_images(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image, label

train_set = train_set.map(read_images)
batch_size = 32
train_set = train_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# run forward prop for all images in the train dataset and concatenate the relevant variables

def get_maps(train_set_ims):
    all_outputs = [meta_model.get_vis_local_maps(image) for image, label in train_set_ims]
    # concat appropriately
    concept_map = tf.concat([x[0] for x in all_outputs], 1)
    print(concept_map.get_shape())
    class_prototype = tf.concat([x[1] for x in all_outputs], 2)
    print(class_prototype.get_shape())
    class_protoweight = tf.concat([x[2] for x in all_outputs], 2)
    class_protoweight = tf.squeeze(class_protoweight)
    print(class_protoweight.get_shape())
    relevance = tf.concat([x[3] for x in all_outputs], 0)
    print(relevance.get_shape())
    return concept_map, class_prototype, class_protoweight, relevance

def create_histograms(c_pw, i, j):
    c_pw_np = c_pw.numpy()
    hist = np.histogram(c_pw_np, 20)
    count = 0
    print(hist[0])
    print(hist[1])
    for index, k in enumerate(hist[1]):
        if k <= 0.05:
            count += hist[0][index]
    return 3968-count
    
def get_positive_count(c_pw, i, j, wei):
    c_pw_np = c_pw.numpy()
    z = c_pw_np.tolist()
    count = 0
    for a in z:
        if a > wei:
            count += 1
    return count

concept_maps, class_prototypes, class_protoweights, relevances = get_maps(train_set)

def get_weights(layer_weights, prototype, i, j):
    weights = layer_weights[i].weights[0]
    a = tf.split(weights, [32 for i in range(10)], 0)
    return tf.squeeze(tf.matmul(prototype, a[j]),  1)
    

class_list = ['fox', 'german+shepherd', 'horse', 'leopard', 'lion', 'persian+cat', 'siamese+cat', 'tiger', 'wolf', 'zebra']

stats = np.zeros((10, 10, 3))
           
def visualize():
    k1 = 100
    k2 = 10
    layer_weights = meta_model.interpret_layer.get_relevance_part()
    # loop for every class
    for i in range(10):
        relv = relevances[:,i]
        for j in range(10):
            c_m = concept_maps[ i*10 + j ]
            c_pw = class_protoweights[i][j]
            c_prototype = class_prototypes[i][j]
            new_c_pw = get_weights(layer_weights, c_prototype, i ,j)
            create_histograms(new_c_pw, i, j)
            weight_rel = new_c_pw
            values,indexes = tf.nn.top_k( weight_rel , k = k2 )
            cm = tf.gather( c_m , indexes , axis = 0 )
            names = [train_images[x] for x in indexes]
            
            for k in range(k2):
                image = names[k]
                image = tf.io.read_file(image)
                image = tf.image.decode_jpeg(image)
                image = tf.image.convert_image_dtype(image, tf.float32)
                image = tf.image.resize(image, [224, 224])
                cmi = cm[k]
                cmi = tf.expand_dims(cmi,-1)
                map_img = tf.image.resize(cmi, [224,224])
                map_img = map_img*image
                map_img = map_img/tf.reduce_max(map_img)
                plt.imsave(dir_v + '/' + str(i) + '_' + str(j) + '_' + str(k) + '_m_' + '.png' , map_img.numpy())
                plt.imsave(dir_v + '/' + str(i) + '_' + str(j) + '_' + str(k) + '_i_' + '.png' , image.numpy())

visualize()

print(consider_concept)

