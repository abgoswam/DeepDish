
# coding: utf-8

# In[1]:

import DataUtils as du
import numpy as np
import tensorflow as tf
import math
import timeit
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[2]:

print(du.images_cooked_train_metadata_filename)


# In[3]:

df = pd.read_csv(du.images_cooked_train_metadata_filename, sep=',', header=None, names=['label', 'label_name', 'img'])
df['label_name'].value_counts()

# In[4]:

X_train, y_train, P_train, X_val, y_val, P_val, X_test, y_test, P_test = du.loadData()


# Normalize the data: subtract the mean image
mean_image = np.mean(X_train, axis=0, dtype=np.uint8)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# In[ ]:

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

classes = ['burger', 'fries', 'burrito', 'lasagna', 'pizza', 'pasta', 'biryani', 'sushi']
num_classes = len(classes)
samples_per_class = 7
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

#plt.savefig('foo2.png')
plt.show()
#
## Visualize some examples from the dataset.
## We show a few examples of training images from each class.
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#num_classes = len(classes)
#samples_per_class = 7
#for y, cls in enumerate(classes):
#    idxs = np.flatnonzero(y_train == y)
#    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(X_train[idx].astype('uint8'))
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)
#plt.show()



## In[5]:
#
## clear old variables
#tf.reset_default_graph()
#
## setup input (e.g. the data that changes every batch)
## The first dim is None, and gets sets automatically based on batch size fed in
#X = tf.placeholder(tf.float32, [None, 32, 32, 3])
#y = tf.placeholder(tf.int64, [None])
#is_training = tf.placeholder(tf.bool)
#
#def simple_model(X,y):
#    # define our weights (e.g. init_two_layer_convnet)
#    
#    # setup variables
#    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
#    bconv1 = tf.get_variable("bconv1", shape=[32])
#    W1 = tf.get_variable("W1", shape=[5408, 10])
#    b1 = tf.get_variable("b1", shape=[10])
#
#    # define our graph (e.g. two_layer_convnet)
#    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
#    h1 = tf.nn.relu(a1)
#    h1_flat = tf.reshape(h1,[-1,5408])
#    y_out = tf.matmul(h1_flat,W1) + b1
#    return y_out
#
#y_out = simple_model(X,y)
#
## define our loss
#total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
#mean_loss = tf.reduce_mean(total_loss)
#
## define our optimizer
#optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
#train_step = optimizer.minimize(mean_loss)
#
#
## In[6]:
#
#def run_model(session, predict, loss_val, Xd, yd,
#              epochs=1, batch_size=64, print_every=100,
#              training=None, plot_losses=False):
#    # have tensorflow compute accuracy
#    correct_prediction = tf.equal(tf.argmax(predict,1), y)
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    
#    # shuffle indicies
#    train_indicies = np.arange(Xd.shape[0])
#    np.random.shuffle(train_indicies)
#
#    training_now = training is not None
#    
#    # setting up variables we want to compute (and optimizing)
#    # if we have a training function, add that to things we compute
#    variables = [mean_loss,correct_prediction,accuracy]
#    if training_now:
#        variables[-1] = training
#    
#    # counter 
#    iter_cnt = 0
#    for e in range(epochs):
#        # keep track of losses and accuracy
#        correct = 0
#        losses = []
#        # make sure we iterate over the dataset once
#        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
#            # generate indicies for the batch
#            # (abgoswam) start_idx = (i*batch_size)%X_train.shape[0]
#            start_idx = (i*batch_size)%Xd.shape[0]
#            
#            idx = train_indicies[start_idx:start_idx+batch_size]
#            
#            # create a feed dictionary for this batch
#            feed_dict = {X: Xd[idx,:],
#                         y: yd[idx],
#                         is_training: training_now }
#            # get batch size
#            # (abgoswam) actual_batch_size = yd[i:i+batch_size].shape[0]
#            actual_batch_size = yd[idx].shape[0]
#            
#            # have tensorflow compute loss and correct predictions
#            # and (if given) perform a training step
#            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
#            
#            # aggregate performance stats
#            losses.append(loss*actual_batch_size)
#            correct += np.sum(corr)
#            
#            # print every now and then
#            if training_now and (iter_cnt % print_every) == 0:
#                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
#            iter_cnt += 1
#        total_correct = correct/Xd.shape[0]
#        total_loss = np.sum(losses)/Xd.shape[0]
#        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"              .format(total_loss,total_correct,e+1))
#        if plot_losses:
#            plt.plot(losses)
#            plt.grid(True)
#            plt.title('Epoch {} Loss'.format(e+1))
#            plt.xlabel('minibatch number')
#            plt.ylabel('minibatch loss')
#            plt.show()
#    return total_loss,total_correct
#
#with tf.Session() as sess:
#    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
#        sess.run(tf.global_variables_initializer())
#        print('Training')
#        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
#        print('Validation')
#        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
#
#
## In[7]:
#
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#
#print('Test')
#run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
#
#
## In[ ]:
#


