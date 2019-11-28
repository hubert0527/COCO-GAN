import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import snconv2d, snlinear
from models.model_base import Model

_EPS = 1e-5

class DiscriminatorBuilder(Model):
    def __init__(self, config):
        self.config=config
        self.ndf_base = self.config["model_params"]["ndf_base"]
        self.num_extra_layers = self.config["model_params"]["d_extra_layers"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]

        self.update_collection = "D_update_collection"

    def _d_residual_block(self, x, out_ch, idx, is_training, resize=True, is_head=False):
        update_collection = self._get_update_collection(is_training)
        with tf.variable_scope("d_resblock_"+str(idx), reuse=tf.AUTO_REUSE):
            h = x
            if not is_head:
                h = tf.nn.relu(h)
            h = snconv2d(h, out_ch, name='d_resblock_conv_1', update_collection=update_collection)
            h = tf.nn.relu(h)
            h = snconv2d(h, out_ch, name='d_resblock_conv_2', update_collection=update_collection)
            if resize:
                h = slim.avg_pool2d(h, [2, 2])

            # Short cut
            s = x
            if resize:
                s = slim.avg_pool2d(s, [2, 2])
            s = snconv2d(s, out_ch, k_h=1, k_w=1, name='d_resblock_conv_sc', update_collection=update_collection)
            return h + s
    
            
    def forward(self, x, y=None, is_training=True):
        valid_sizes = {8, 16, 32, 64, 128, 256, 512}
        assert (self.macro_patch_size[0] in valid_sizes and self.macro_patch_size[1] in valid_sizes), \
            "I haven't test your macro patch size: {}".format(self.macro_patch_size)

        update_collection = self._get_update_collection(is_training)
        print(" [Build] Discriminator ; is_training: {}".format(is_training))
        
        with tf.variable_scope("D_discriminator", reuse=tf.AUTO_REUSE):

            num_resize_layers = int(math.log(min(self.macro_patch_size), 2) - 1)
            num_total_layers  = num_resize_layers + self.num_extra_layers
            basic_layers = [2, 4, 8, 8]
            if num_total_layers>len(basic_layers):
                num_replicate_layers = num_total_layers - len(basic_layers)
                ndf_mult_list = [1, ] * num_replicate_layers + basic_layers
            else:
                ndf_mult_list = basic_layers[-num_total_layers:]
                ndf_mult_list[0] = 1
            print("\t ndf_mult_list = {}".format(ndf_mult_list))

            # Stack extra layers without resize first
            h = x
            for idx, ndf_mult in enumerate(ndf_mult_list):
                n_ch = self.ndf_base * ndf_mult
                # Head is fixed and goes first
                if idx==0:
                    is_head, resize, is_extra = True, True, False
                # Extra layers before standard layers
                elif idx<=self.num_extra_layers:
                    is_head, resize, is_extra = False, False, True
                # Last standard layer has no resize
                elif idx==len(ndf_mult_list)-1:
                    is_head, resize, is_extra = False, False, False
                # Standard layers
                else:
                    is_head, resize, is_extra = False, True, False
                
                h = self._d_residual_block(h, n_ch, idx=idx, is_training=is_training, resize=resize, is_head=is_head)
                print("\t DResBlock: id={}, out_shape={}, resize={}, is_extra={}"
                    .format(idx, h.shape.as_list(), resize, is_extra))

            h = tf.nn.relu(h)
            h = tf.reduce_sum(h, axis=[1,2]) # Global pooling
            last_feature_map = h
            adv_out = snlinear(h, 1, 'main_steam_out', update_collection=update_collection)

            # Projection Discriminator
            if y is not None:
                h_num_ch = self.ndf_base*ndf_mult_list[-1]
                y_emb = snlinear(y, h_num_ch, 'y_emb', update_collection=update_collection)
                proj_out = tf.reduce_sum(y_emb*h, axis=1, keepdims=True)
            else:
                proj_out = 0

            out = adv_out + proj_out
            
            return out, last_feature_map

