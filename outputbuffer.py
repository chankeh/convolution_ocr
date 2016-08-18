from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from PIL import Image
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 256
TRAIN_SIZE = 1000   #one word
TEST_SIZE = 200     #one word
TRAINING_DIRECTORY = 'trainingdata/img2'
TESTING_DIRECTORY = 'testingdata/img2'

def load_data(filename,num_images,size):
    #Return a new array of given shape and type, without initializing entries.
    data = np.empty((num_images,IMAGE_SIZE,IMAGE_SIZE,1),dtype='uint8')
    label = np.empty((num_images,),dtype='uint16')
    datadisorder = np.empty((num_images,IMAGE_SIZE,IMAGE_SIZE,1),dtype='uint8');
    labeldisorder = np.empty((num_images,), dtype='uint16')
    #dirs = os.listdir('training data/img2')
    dirs = os.listdir(filename)
    numofdirs = len(dirs)
    #numofdirs = num_images//size
    for i in range(numofdirs):
        filepath = os.path.join(filename, dirs[i])
        imgs =  os.listdir(filepath)
        numofimgs = len(imgs)
        for j in range(numofimgs):
            img = Image.open(os.path.join(filepath, imgs[j])).convert('L')
            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            arr = np.asarray(img,dtype='uint8')
            #arr = (arr - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            data[i*size+j,:,:,0] = arr
            label[i*size+j] = i #int(dirs[i],16)
            #print hex(label[i*1000+j])
    arraylist = np.arange(num_images)
    random.shuffle(arraylist)
    random.shuffle(arraylist)
    for i in range(num_images):
        datadisorder[i,...]  = data[arraylist[i],...]
        labeldisorder[i,...] = label[arraylist[i],...]
    return datadisorder,labeldisorder
train_data, train_labels = load_data(TRAINING_DIRECTORY, TRAIN_SIZE * NUM_LABELS, 300)
train_data.tofile("train_image_u8.data")   #output buffer
train_labels.tofile("train_label_u16.data")  #output buffer
test_data, test_labels = load_data(TESTING_DIRECTORY, TEST_SIZE * NUM_LABELS, 50)
test_data.tofile("test_image_u8.data")  #output buffer
test_labels.tofile("test_label_u16.data")  #output buffer
print ("ok")
