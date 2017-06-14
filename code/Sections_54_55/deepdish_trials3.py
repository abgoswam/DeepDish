# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 22:44:21 2017

@author: agoswami
"""

#######
import matplotlib.pyplot as plt
import DataUtils as du
import numpy as np

X_train, y_train, P_train, X_val, y_val, P_val, X_test, y_test, P_test = du.loadData()

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

classes = ['burger', 'fries', 'burrito', 'lasagna', 'pizza', 'pasta', 'biryani', 'sushi']
num_classes = len(classes)
samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)

plt.show()
mean_image = np.mean(X_train, axis=0)
plt.figure(figsize=(4,4))
plt.imshow(mean_image.astype('uint8')) # visualize the mean image
plt.show()

## second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
#######

import tensorflow as tf
import matplotlib.pyplot as plt

##### UTILS #####

# Create the model
x = tf.placeholder(tf.float32, [None, 32, 32, 3])

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [None])

# Placeholders for batchnorm and dropout
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def bn_conv2d(x, W, is_training):
  """conv2d returns a 2d convolution layer with full stride."""
  x_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  x_bn_conv = tf.contrib.layers.batch_norm(
        x_conv, 
        decay=0.99,
        scale=True,  
        center=True,  
        is_training=is_training, 
        updates_collections=None )

  return x_bn_conv

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def get_batch(x, y, batch_size):
#    print("x shape : {0}".format(x.shape))
#    print("y shape : {0}".format(y.shape))
    n_samples = x.shape[0]
    indices = np.random.choice(n_samples, batch_size)
    return x[indices], y[indices], indices
    

##### MODEL #####

# First convolutional layer - maps one grayscale image to 32 feature maps.
#W_conv1 = weight_variable([5, 5, 3, 32])
#b_conv1 = bias_variable([32])
W_conv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32])
b_conv1 = tf.get_variable("bconv1", shape=[32])
h_conv1 = tf.nn.relu(bn_conv2d(x, W_conv1, is_training) + b_conv1)

W_conv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 32])
b_conv2 = tf.get_variable("bconv2", shape=[32])
h_conv2 = tf.nn.relu(bn_conv2d(h_conv1, W_conv2, is_training) + b_conv2)

W_conv3 = tf.get_variable("Wconv3", shape=[5, 5, 32, 32])
b_conv3 = tf.get_variable("bconv3", shape=[32])
h_conv3 = tf.nn.relu(bn_conv2d(h_conv2, W_conv3, is_training) + b_conv3)

W_conv4 = tf.get_variable("Wconv4", shape=[5, 5, 32, 32])
b_conv4 = tf.get_variable("bconv4", shape=[32])
h_conv4 = tf.nn.relu(bn_conv2d(h_conv3, W_conv4, is_training) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = tf.get_variable("Wconv5", shape=[5, 5, 32, 32])
b_conv5 = tf.get_variable("bconv5", shape=[32])
h_conv5 = tf.nn.relu(bn_conv2d(h_pool4, W_conv5, is_training) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

W_fc1 = tf.get_variable("W1", shape=[8 * 8 * 32, 1024])
b_fc1 = tf.get_variable("b1", shape=[1024])

h_pool5_flat = tf.reshape(h_pool5, [-1, 8 * 8 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.get_variable("W2", shape=[1024, 20])
b_fc2 = tf.get_variable("b2", shape=[20])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##### TRAIN and EVALUATE ####
                  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_, depth=20), logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_history = []
train_accuracy_history = []
val_accuracy_history = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = get_batch(X_train, y_train, 50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, is_training:False})
            val_accuracy = accuracy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0, is_training:False})
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)
            print('step %d, training accuracy %g, , validation accuracy %g' % (i, train_accuracy, val_accuracy))
                  
        _, loss_i = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, is_training:True})
        loss_history.append(loss_i)

    print('test accuracy %g' % accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0, is_training:False}))
    
# Run this cell to visualize training loss and train / val accuracy

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(train_accuracy_history, '-o', label='train')
plt.plot(val_accuracy_history, '-o', label='val')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()