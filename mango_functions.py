from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import os




# pylint: disable=unused-import
import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


import tensorflow as tf

def weight_variable(shape):
    inital_aaaa = tf.truncated_normal(shape, stddev=0.1)
    inital = inital_aaaa/1.0
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')