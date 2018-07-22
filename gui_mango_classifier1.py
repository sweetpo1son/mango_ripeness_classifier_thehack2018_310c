from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import os

import tkinter as tk
'''from out import *'''
from tkinter import filedialog
from PIL import Image,ImageTk






import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import tensorflow as tf
from mango_functions import *
def out(path):
    print ('starting to test!'+path)
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


    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    save_path = saver.restore(sess, "model.ckpt")
    print(sess.run(W_conv1))


    test_x=numpy.zeros((1,76800))
    test_y=numpy.zeros((1,3))

    img = Image.open(path)
    img = img.resize((160, 160))
    img_array = numpy.array(img)
    cnt = 0

    for d in range(img_array.__len__()):
        for c in range(img_array[0].__len__()):
            for a in range(img_array[0][0].__len__()):
                test_x[0][cnt] = img_array[d][c][a]/255.0-0.5
                cnt+=1


    y_out =sess.run(y, feed_dict={x: test_x, y_: test_y})
    res=[]
    for i in range(3):
        y_out[0][i]=round(y_out[0][i],4)

    res.append('underripe: '+str(y_out[0][0]))
    res.append('ripe: '+str(y_out[0][1]))
    res.append('overripe: '+str(y_out[0][2]))

    if max(y_out[0])==y_out[0][0]:
        return ['underripe']+res
    else:
        if max(y_out[0])==y_out[0][1]:
            return ['ripe']+res
        else:
            return ['overripe']+res

    
window = tk.Tk()
window.title('Mango ripeness classifier')
window.geometry('1000x500')


path = tk.StringVar()

def choosepic():
    path_=filedialog.askopenfilename()
    path.set(path_)
    img_open = Image.open(e1.get())
    im = img_open.resize((160, 160))
    img=ImageTk.PhotoImage(im)
    l1.config(image=img)
    l1.image=img

tk.Button(window,text='choose mango image',command=choosepic).pack()
e1=tk.Entry(window,state='readonly',text=path)
e1.pack()
l1=tk.Label(window,width=200,height=200)
l1.place(x=100, y=100, anchor='nw')



var2 = tk.StringVar()
var2.set('waiting....')  
entry = tk.Entry(window,textvariable = var2).place(x=400, y=200, anchor='nw')

var3 = tk.StringVar()
var3.set('waiting....')  
entry3 = tk.Entry(window,textvariable = var3).place(x=400, y=300, anchor='nw')

var4 = tk.StringVar()
var4.set('waiting....')  
entry4 = tk.Entry(window,textvariable = var4).place(x=800, y=300, anchor='nw')

var5 = tk.StringVar()
var5.set('waiting....')  
entry5 = tk.Entry(window,textvariable = var5).place(x=600, y=300, anchor='nw')

def moveit1():
    res=out(e1.get())
    print(res)
    var2.set(res[0])
    var3.set(res[1])
    var4.set(res[2])
    var5.set(res[3])
b1 = tk.Button(window, text='compute', command=moveit1).place(x=200, y=10, anchor='nw')

window.mainloop()

