
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

import DataUtils as du
X_train, y_train, P_train, X_val, y_val, P_val, X_test, y_test, P_test = du.loadData()

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# In[3]:

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
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

#plt.savefig('assets/data_vis_2.png')
plt.show()


# In[4]:

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
#print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.astype('uint8')) # visualize the mean image
plt.show()


# In[5]:

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

n_samples = X_train.shape[0]
batch_size = 100

# In[ ]:

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)

##### UTILS #####

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

##### MODEL #####

## 1st conv layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## 2nd conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## Dense layer    
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout before readout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##readout layer
W_fc2 = weight_variable([1024, 20])
b_fc2 = bias_variable([20])
y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##### TRAIN and EVALUATE ####
#print("Total parameters in current model : {0}".format(count_params()))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,20), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

# have tensorflow compute accuracy
correct_prediction = tf.equal(tf.argmax(y_out,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_acc_history = []
val_acc_history = []

for i in range(10000):    
    # Select random minibatch
    indices = np.random.choice(n_samples, batch_size)
    
#    indices = np.arange(len(X_data))
    X_batch, y_batch = X_train[indices], y_train[indices]
    
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch, keep_prob: 1.0})
        val_accuracy = sess.run(accuracy, feed_dict={X: X_val, y: y_val, keep_prob: 1.0})
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)
        print("step %d, training accuracy %g, val accuracy %g"%(i, train_accuracy, val_accuracy))
    
        # one step of optimization   
        sess.run([mean_loss, train_step], feed_dict={X: X_batch, y: y_batch, keep_prob: 0.5})


#plt.subplot(2, 1, 2)
#plt.title('Accuracy')
#plt.plot(train_acc_history, '-o', label='train')
#plt.plot(val_acc_history, '-o', label='val')
##plt.plot([0.5] * len(solver.val_acc_history), 'k--')
#plt.xlabel('Epoch')
#plt.legend(loc='lower right')
#plt.gcf().set_size_inches(15, 12)
#plt.show()

print("test accuracy %g"% sess.run(accuracy, feed_dict={X: X_test, y: y_test, keep_prob: 1.0}))


    
    
    
    

