# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:36:58 2017

@author: agoswami
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

# Define input data
X_data_orig = np.arange(100, step=.1)
y_data_orig = X_data_orig + 20 * np.sin(X_data_orig/10)
# Plot input data
plt.scatter(X_data_orig, y_data_orig)

# Define data size and batch size
n_samples = 1000
batch_size = 100

# Tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data_orig, (n_samples,1))
y_data = np.reshape(y_data_orig, (n_samples,1))

# Define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1)) 

# Define variables to be learned
with tf.variable_scope("linear-regression"):
    W = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer(0.0))
    
    y_pred = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y - y_pred)**2/ batch_size)
    
opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)

sess.run(tf.initialize_all_variables())

# Gradient descent loop for 500 steps
for iloop in range(5000):
    # Select random minibatch
    indices = np.random.choice(n_samples, batch_size)
    
#    indices = np.arange(len(X_data))
    X_batch, y_batch = X_data[indices], y_data[indices]
 
    # Do gradient descent step
    opt_opresult, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})
    print("{0},{1}".format(iloop, loss_val))     
    
###################

W_np = sess.run(W)
b_np = sess.run(b)

W_plot = W_np.reshape(1,)[0]
b_plot = b_np.reshape(1,)[0]

print("{0},{1}".format(W_plot, b_plot))

fit = np.array([W_plot, b_plot])
fit_fn = np.poly1d(fit) 

plt.plot(X_data_orig, y_data_orig, 'yo', X_data_orig, fit_fn(X_data_orig), '--k')
plt.show()

