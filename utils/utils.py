import glob
import os
import numpy as np

def create_dataset(dataset_dir = 'datasets/awa2', split = 0.8):
    classes, labels = [], []
    # GET CLASSES AND LABELS
    with open('%s/classes.txt' % dataset_dir) as fp:
        for entry in fp.readlines():
            attrib = entry.strip().split(':')
            labels.append(attrib[0])
            classes.append(attrib[1])

    dataset = []
    for x,y in zip(classes, labels):
        image_dir = '%s/%s/*'%(dataset_dir, x)
        for filename in glob.glob(image_dir):
            entry = '%s:%s'%(y, filename)
            dataset.append(entry)

    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    total_images = dataset.shape[0]
    train_count = int(split*total_images)

    train_set = dataset[:train_count]
    test_set = dataset[train_count:]

    np.savetxt('%s/train.txt'%dataset_dir, train_set, fmt='%s')
    np.savetxt('%s/test.txt'%dataset_dir, test_set, fmt='%s')
        
def load_dataset(dataset_dir = 'datasets/awa2'):
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    with open('%s/train.txt' % dataset_dir) as fp:
        for entry in fp.readlines():
            attrib = entry.strip().split(':')
            train_labels.append(int(attrib[0]))
            train_images.append(attrib[1])

    with open('%s/test.txt' % dataset_dir) as fp:
        for entry in fp.readlines():
            attrib = entry.strip().split(':')
            test_labels.append(int(attrib[0]))
            test_images.append(attrib[1])

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)