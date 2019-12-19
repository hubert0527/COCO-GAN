import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from ops import batch_norm, snconv2d, snlinear, upscale, NO_OPS
from models.model_base import Model

_EPS = 1e-5

class GeneratorBuilder(Model):
    def __init__(self, config):
        self.config=config
        self.ngf_base = self.config["model_params"]["ngf_base"]
        self.num_extra_layers = self.config["model_params"]["g_extra_layers"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.c_dim = self.config["data_params"]["c_dim"]
        self.update_collection = "G_update_collection"

    def _cbn(self, x, y, is_training, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            ch = x.shape.as_list()[-1]
            gamma = slim.fully_connected(y, ch, activation_fn=None)
            beta = slim.fully_connected(y, ch, activation_fn=None)

            mean_rec = tf.get_variable("mean_recorder", [ch,], 
                initializer=tf.constant_initializer(np.zeros(ch)), trainable=False)
            var_rec  = tf.get_variable("var_recorder", [ch,], 
                initializer=tf.constant_initializer(np.ones(ch)), trainable=False)
            running_mean, running_var = tf.nn.moments(x, axes=[0, 1, 2])
            
            if is_training:
                new_mean_rec = 0.99 * mean_rec + 0.01 * running_mean
                new_var_rec  = 0.99 * var_rec  + 0.01 * running_var
                assign_mean_op = mean_rec.assign(new_mean_rec)
                assign_var_op  = var_rec.assign(new_var_rec)
                tf.add_to_collection(self.update_collection, assign_mean_op)
                tf.add_to_collection(self.update_collection, assign_var_op)
                mean = running_mean
                var  = running_var
            else:
                mean = mean_rec
                var  = var_rec

            # tiled_mean = tf.tile(tf.expand_dims(mean, 0), [tf.shape(x)[0], 1])
            # tiled_var  = tf.tile(tf.expand_dims(var , 0), [tf.shape(x)[0], 1])
            mean  = tf.reshape(mean, [1, 1, 1, ch])
            var   = tf.reshape(var , [1, 1, 1, ch])
            gamma = tf.reshape(gamma, [-1, 1, 1, ch])
            beta  = tf.reshape(beta , [-1, 1, 1, ch])
            
            h = (x-mean) / tf.sqrt(var**2+_EPS) 
            return h * gamma + beta

    def _g_residual_block(self, x, y, n_ch, idx, is_training, resize=True):
        update_collection = self._get_update_collection(is_training)
        with tf.variable_scope("g_resblock_"+str(idx), reuse=tf.AUTO_REUSE):
            h = self._cbn(x, y, is_training, scope='g_resblock_cbn_1')
            h = tf.nn.relu(h)
            if resize:
                h = upscale(h, 2)
            h = snconv2d(h, n_ch, name='g_resblock_conv_1', update_collection=update_collection)
            h = self._cbn(h, y, is_training, scope='g_resblock_cbn_2')
            h = tf.nn.relu(h)
            h = snconv2d(h, n_ch, name='g_resblock_conv_2', update_collection=update_collection)

            if resize:
                sc = upscale(x, 2)
            else:
                sc = x
            sc = snconv2d(sc, n_ch, k_h=1, k_w=1, name='g_resblock_conv_sc', update_collection=update_collection)

            return h + sc

    def forward(self, z, coord, is_training):
        valid_sizes = {4, 8, 16, 32, 64, 128, 256}
        assert (self.micro_patch_size[0] in valid_sizes and self.micro_patch_size[1] in valid_sizes), \
            "I haven't test your micro patch size: {}".format(self.micro_patch_size)

        update_collection = self._get_update_collection(is_training)
        print(" [Build] Generator ; is_training: {}".format(is_training))
        
        with tf.variable_scope("G_generator", reuse=tf.AUTO_REUSE):
            init_sp = 2
            init_ngf_mult = 16
            cond = tf.concat([z, coord], axis=1)
            h = snlinear(cond, self.ngf_base*init_ngf_mult*init_sp*init_sp, 'g_z_fc', update_collection=update_collection)
            h = tf.reshape(h, [-1, init_sp, init_sp, self.ngf_base*init_ngf_mult])

            # Stacking residual blocks
            num_resize_layers = int(math.log(min(self.micro_patch_size), 2) - 1)
            num_total_layers  = num_resize_layers + self.num_extra_layers
            basic_layers = [8, 4, 2] 
            if num_total_layers>=len(basic_layers):
                num_replicate_layers = num_total_layers - len(basic_layers)
                ngf_mult_list = basic_layers + [1, ] * num_replicate_layers
            else:
                ngf_mult_list = basic_layers[:num_total_layers]
            print("\t ngf_mult_list = {}".format(ngf_mult_list))

            for idx, ngf_mult in enumerate(ngf_mult_list):
                n_ch = self.ngf_base * ngf_mult
                # Standard layers first
                if idx < num_resize_layers:
                    resize, is_extra = True, False
                # Extra layers do not resize spatial size
                else:
                    resize, is_extra = False, True
                h = self._g_residual_block(h, cond, n_ch, idx=idx, is_training=is_training, resize=resize)
                print("\t GResBlock: id={}, out_shape={}, resize={}, is_extra={}"
                    .format(idx, h.shape.as_list(), resize, is_extra))

            h = batch_norm(name="g_last_bn")(h, is_training=is_training)
            h = tf.nn.relu(h)
            h = snconv2d(h, self.c_dim, name='g_last_conv_2', update_collection=update_collection)
            return tf.nn.tanh(h)
