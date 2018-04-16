import math
import numpy as np
import tensorflow as tf


def batch_norm(x, phase, name):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        n_out = x.shape[3]
        beta = tf.get_variable('beta', [n_out], initializer = tf.constant_initializer(0.0), trainable = True)
        gamma = tf.get_variable('gamma', [n_out], initializer = tf.constant_initializer(1.0), trainable = True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name = 'bn')

    return normed


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def conv_layer(input_, filters, filter_size, stride, phase, kernel_reg, kernel_init, h_layer, relu_slope=0.0):
    h = tf.layers.conv2d(input_, filters, (filter_size, filter_size), (stride, stride), 'SAME',
                         kernel_regularizer=kernel_reg, kernel_initializer=kernel_init, name='conv{}'.format(h_layer))
    h = tf.layers.batch_normalization(h, name='batch{}'.format(h_layer), momentum=0.0, epsilon=1e-5, training=phase, fused=True)
    h = tf.nn.leaky_relu(h, alpha=relu_slope, name='relu{}'.format(h_layer))
    return h