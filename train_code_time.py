# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 22:38:16 2018

@author: Shilong_Wang
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

data_test = xlrd.open_workbook('D:\\ECIT\\Masks\\test_data.xlsx')
table_test = data_test.sheets()[0]
test_nrows = table_test.nrows #number of rows
test_ncols = table_test.ncols #number of columns

test_datamatrix=np.zeros((test_nrows,test_ncols))

for x in range(test_ncols):
    test_cols =table_test.col_values(x)    
    
    test_cols1=np.matrix(test_cols)

    test_datamatrix[:,x]=test_cols1
    
    
species_test=np.zeros((test_nrows,1))
species_test=test_datamatrix[:,0]-1
temperature_test=np.zeros((test_nrows,1))
temperature_test=test_datamatrix[:,1]-1
time_test=np.zeros((test_nrows,1))
time_test=test_datamatrix[:,2]-1
dye_test=np.zeros((test_nrows,1))
dye_test=test_datamatrix[:,3]-1
y_species_test=tf.one_hot(species_test,4,on_value=1,off_value=None,axis=1)
y_temperature_test=tf.one_hot(temperature_test,2,on_value=1,off_value=None,axis=1)
y_time_test=tf.one_hot(time_test,4,on_value=1,off_value=None,axis=1)
y_dye_test=tf.one_hot(dye_test,23,on_value=1,off_value=None,axis=1)
#y_test_tf=tf.concat([y_species_test, y_temperature_test, y_time_test, y_dye_test], 1)
#y_test_tf=tf.concat([y_species_test, y_temperature_test, y_time_test, y_dye_test], 1)
with tf.Session()as sess:
    y_test = y_time_test.eval()
    #print(sess.run(species_test))
    print(sess.run(y_dye_test))

x_test=np.zeros((test_nrows,3))
x_test=test_datamatrix[:,4:7]


data_train = xlrd.open_workbook('D:\\ECIT\\Masks\\train_data.xlsx')
table_train = data_train.sheets()[0]
train_nrows = table_train.nrows #number of rows
train_ncols = table_train.ncols #number of columns

train_datamatrix=np.zeros((train_nrows,train_ncols))
for x in range(train_ncols):
    train_cols =table_train.col_values(x)    
    
    train_cols1=np.matrix(train_cols)

    train_datamatrix[:,x]=train_cols1
species_train=np.zeros((train_nrows,1))
species_train=train_datamatrix[:,0]-1
temperature_train=np.zeros((train_nrows,1))
temperature_train=train_datamatrix[:,1]-1
time_train=np.zeros((train_nrows,1))
time_train=train_datamatrix[:,2]-1
dye_train=np.zeros((train_nrows,1))
dye_train=train_datamatrix[:,3]-1
y_species_train=tf.one_hot(species_train,4,on_value=1,off_value=None,axis=1)
y_temperature_train=tf.one_hot(temperature_train,2,on_value=1,off_value=None,axis=1)
y_time_train=tf.one_hot(time_train,4,on_value=1,off_value=None,axis=1)
y_dye_train=tf.one_hot(dye_train,23,on_value=1,off_value=None,axis=1)
#y_train_tf=tf.concat([y_species_train, y_temperature_train, y_time_train, y_dye_train], 1)
with tf.Session()as sess:
    y_train = y_time_train.eval()
    #print(sess.run(species_train))
    print(sess.run(y_dye_train))
x_train=np.zeros((train_nrows,3))
x_train=train_datamatrix[:,4:7]
    
    


# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 30

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 32 # 1st layer num features
num_code=64
num_hidden_2 = 64 # 2nd layer num features (the latent dim)
num_input = 3 # MNIST data input (img shape: 28*28)·
#alpha=0.0001
num_output = 4

train_loss=np.zeros((num_steps//10,1))
test_loss=np.zeros((num_steps//10,1))


def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)



with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,num_input],name='x_input')
    y=tf.placeholder(tf.float32,[None,num_output],name='y_input')
with tf.name_scope('hidden_1'):
    w1=weight_variable([num_input,num_hidden_1],name='w1')
    b1=bias_variable([num_hidden_1],name='b1')
    with tf.name_scope('node_1'):
        node_1=tf.matmul(x,w1)+b1
    with tf.name_scope('relu'):
        h_1=tf.nn.relu(node_1)


with tf.name_scope('encode'):
    w=weight_variable([num_hidden_1,num_code],name='w')
    b=bias_variable([num_code],name='b')
    with tf.name_scope('sum_encode'):
        sum_encode=tf.matmul(h_1,w)+b
    with tf.name_scope('relu'):
        h_encode=tf.nn.relu(sum_encode)

with tf.name_scope('decode'):
    w=weight_variable([num_code,num_hidden_2],name='w')
    b=bias_variable([num_hidden_2],name='b')
    with tf.name_scope('sum_decode'):
        sum_decode=tf.matmul(h_encode,w)+b
    with tf.name_scope('relu'):
        h_decode=tf.nn.relu(sum_decode)


with tf.name_scope('hidden_2'):
    
    w1=weight_variable([num_hidden_2,num_output],name='w1')
    b1=bias_variable([num_output],name='b1')
    with tf.name_scope('node_1'):
        node_1=tf.matmul(h_decode,w1)+b1
    with tf.name_scope('relu'):
        h_2=tf.nn.relu(node_1)
    with tf.name_scope('prediction'):
        prediction=tf.nn.softmax(h_2)

with tf.name_scope('loss_mean_square'):
    #loss_mean_square=tf.reduce_mean(tf.pow(y-h_2,2))
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    #tf.summary.scalar('loss_mean_square',loss_mean_square)
    tf.summary.scalar('cross',cross_entropy)
with tf.name_scope('train'):

    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #train_step=tf.train.RMSPropOptimizer(learning_rate).minimize(loss_mean_square)
    
    
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction= tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)

    batch_size=30
    batch_count=int(train_nrows/batch_size)
    reminder=train_nrows%batch_size
    for i in range(num_steps):
        for n in range(batch_count):
            
            train_step.run(feed_dict={x: x_train[n*batch_size:(n+1)*batch_size], y: y_train[n*batch_size:(n+1)*batch_size]})  

        if reminder>0:
            start_index = batch_count * batch_size;  
            train_step.run(feed_dict={x: x_train[start_index:train_nrows-1], y: y_train[start_index:train_nrows-1]})  
        
        iterate_accuracy = 0 
        if i%10==0:
            train_loss[i//10,0]=sess.run(accuracy,feed_dict={x:x_train,y:y_train})
            test_loss[i//10,0]=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
            print('Iter'+str(i)+', Testing Accuracy= '+str(test_loss[i//10,0])+',Training Accuracy=' +str(train_loss[i//10,0]))
 
    x_index = np.linspace(0, num_steps, 100)
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 32,
    }
    figsize = 8,8
    figure, ax = plt.subplots(figsize=figsize)
    #figure, ax=plt.figure(figsize=(8, 8))
    
    A,=plt.plot(x_index, train_loss, color="red",label='train_accuracy',linewidth=2.0,ms=10)
    B,=plt.plot(x_index, test_loss, color="blue",label='test_accuracy',linewidth=2.0,ms=10)
    plt.legend(handles=[A,B],prop=font1)
    plt.xlabel("Interation", font1)
    plt.ylabel("Accuracy (Time)", font1)
    
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.show()
    
    
   