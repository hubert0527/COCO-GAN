"""
Most of the codes are from:
1. https://github.com/carpedm20/DCGAN-tensorflow
2. https://github.com/minhnhat93/tf-SNDCGAN
"""
import math
import warnings
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops


NO_OPS = 'NO_OPS'


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, is_training):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=is_training,
                      scope=self.name)


def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1
    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )
    if update_collection is None:
        warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                    '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0


def snconv2d(input_, output_dim, name="conv2d", k_h=3, k_w=3, d_h=1, d_w=1, spectral_normed=True, 
             stddev=None, update_collection=None, with_w=False, padding="SAME"):
    # Glorot intialization
    # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
    fan_out = k_h * k_w * output_dim
    if stddev is None:
        stddev = np.sqrt(2. / (fan_in))

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectral_normed:
            conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                                strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), [-1,]+conv.get_shape().as_list()[1:])
        if with_w:
            return conv, w, biases
        else:
            return conv
      

def snlinear(input_, output_size, name="linear", spectral_normed=True, 
             stddev=None, bias_start=0.0, with_biases=True,
             update_collection=None, with_w=False, initializer=None):
    shape = input_.get_shape().as_list()

    if stddev is None:
        stddev = np.sqrt(1. / (shape[1]))
    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        
        w_initializer = initializer if initializer==None else tf.truncated_normal_initializer(stddev=stddev)
        weight = tf.get_variable("w", [shape[1], output_size], tf.float32, w_initializer)
        if with_biases:
            bias = tf.get_variable("b", [output_size],
                                initializer=tf.constant_initializer(bias_start))
        if spectral_normed:
            mul = tf.matmul(input_, spectral_normed_weight(weight, update_collection=update_collection))
        else:
            mul = tf.matmul(input_, weight)
        if with_w:
            if with_biases:
                return mul + bias, weight, bias
            else:
                return mul, weight, None
        else:
            if with_biases:
                return mul + bias
            else:
                return mul


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def get_conv_shape(tensor):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    return shape


def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return tf.image.resize_nearest_neighbor(x, (h*scale, w*scale))


def pad(x, p):
    c = tf.constant([[0, 0], [p, p,], [p, p], [0, 0]])
    return tf.pad(x, c, mode='SYMMETRIC')


def add_coords(input_tensor, x_dim=64, y_dim=64, with_r=False):
    """
    For CoordConv.

    Add coords to a tensor
    input_tensor: (batch, x_dim, y_dim, c)
    """
    batch_size_tensor = tf.shape(input_tensor)[0]
    
    xx_ones = tf.ones([batch_size_tensor, x_dim],
        dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0),
        [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, 1)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)
    
    yy_ones = tf.ones([batch_size_tensor, y_dim],
        dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0),
        [batch_size_tensor, 1])
    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)
    
    xx_channel = tf.cast(xx_channel, "float32") / (x_dim - 1)
    yy_channel = tf.cast(yy_channel, "float32") / (y_dim - 1)
    xx_channel = xx_channel*2 - 1
    yy_channel = yy_channel*2 - 1
    
    ret = tf.concat([input_tensor,
        xx_channel,
        yy_channel], axis=-1)
        
    if with_r:
        rr = tf.sqrt( tf.square(xx_channel-0.5)
                + tf.square(yy_channel-0.5)
            )
        ret = tf.concat([ret, rr], axis=-1)
    return ret

