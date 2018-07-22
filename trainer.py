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
from mango_functions import *
print ('starting to train and test!')
x = tf.placeholder(tf.float32, [None, 76800])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x,[-1,160,160,3])
W_conv1 = weight_variable([5,5,3,20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,20,10])
b_conv2 = bias_variable([10])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1,40*40*10])
W_fc2 = weight_variable([40*40*10,50])
b_fc2 = bias_variable([50])
h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc2)+b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

W_fc3 = weight_variable([50,3])
b_fc3 = bias_variable([3])
y = tf.nn.softmax(tf.matmul(h_fc2,W_fc3)+b_fc3)

y_ = tf.placeholder("float", [None,3])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

train_num=100
test_num=30
iter_num=1

print(W_conv1.eval(sess))
for i in range(train_num*iter_num):
  print (i)
  batch_xs=numpy.zeros((3,76800))
  batch_ys=numpy.zeros((3,3))
  for j in range(3):
      batch_ys[j][j]=1
      zz=i//train_num
      yy=i-train_num*zz
      img = Image.open('mango_dataset_resize/' +str(j)+'/'+str(yy+1)+'.jpg')
      img_array = numpy.array(img)
      cnt=0

      for d in range(img_array.__len__()):
          for c in range(img_array[0].__len__()):
              for a in range(img_array[0][0].__len__()):
                  batch_xs[j][cnt]=img_array[d][c][a]/255.0-0.5
                  cnt+=1

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(sess.run(W_conv1))
  
save_path = saver.save(sess, "model.ckpt")
print("data saved in file: %s" % save_path)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
test_x=numpy.zeros((3*test_num,76800))
test_y=numpy.zeros((3*test_num,3))

for i in range(3*test_num):
    m=i//3
    test_y[i][i-3*m]=1
for j in range(3*test_num):
    m=j//3
    r=j-3*m
    img = Image.open('mango_dataset_resize/' + str(r) + '/' + str(m+1) + '.jpg')
    img_array = numpy.array(img)
    cnt = 0

    for d in range(img_array.__len__()):
        for c in range(img_array[0].__len__()):
            for a in range(img_array[0][0].__len__()):
                test_x[j][cnt] = img_array[d][c][a]/255.0-0.5
                cnt+=1


print (sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

