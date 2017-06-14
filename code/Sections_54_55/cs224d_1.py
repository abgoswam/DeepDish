# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf

#tf.InteractiveSession()

sess = tf.Session()

#input1 = tf.placeholder(tf.float32)
#input2 = tf.placeholder(tf.float32)
#output = tf.multiply(input1, input2)
#
#with tf.Session() as sess:
#    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])

assert v1 == v

a = np.array([[1,2,3], [4,5,6]])
ta = tf.convert_to_tensor(a)


#state = tf.Variable(0, name="counter")
#new_value = tf.add(state, tf.constant(1))
#update = tf.assign(state, new_value)
#
#sess.run(tf.global_variables_initializer())
#print(sess.run(state))
#for _ in range(3):
#    sess.run(update)
#    print(sess.run(state))
