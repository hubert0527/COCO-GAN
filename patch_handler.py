import tensorflow as tf
import numpy as np

from math import floor, sqrt, pi
from numpy import sin, cos

class PatchHandler():

    def __init__(self, config):
        self.config = config

        self.batch_size = self.config["train_params"]["batch_size"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]
        self.full_image_size = self.config["data_params"]["full_image_size"]
        self.coordinate_system = self.config["data_params"]["coordinate_system"]
        self.c_dim = self.config["data_params"]["c_dim"]

        self.num_micro_compose_macro = config["data_params"]["num_micro_compose_macro"]


    def reord_patches_cpu(self, x, batch_size, patch_count):
        # Reorganize image order from [a0, b0, c0, a1, b1, c1, ...] to [a0, a1, ..., b0, b1, ..., c0, c1, ...]
        select = np.hstack([[i*batch_size+j for i in range(patch_count)] for j in range(batch_size)])
        x_reord = np.take(x, select, axis=0)
        return x_reord

        
    def reord_patches_gpu(self, x, batch_size, num_patches):
        # Reorganize image order from [a0, b0, c0, a1, b1, c1, ...] to [a0, a1, ..., b0, b1, ..., c0, c1, ...]
        select = np.hstack([[i*batch_size+j for i in range(num_patches)] for j in range(batch_size)])
        x_reord = tf.gather(x, select, axis=0)
        return x_reord


    def concat_micro_patches_cpu(self, generated_patches, ratio_over_micro):

        patch_count = ratio_over_micro[0] * ratio_over_micro[1]
        generated_patches = np.concatenate(generated_patches, axis=0)

        stage1_shape = [
            -1, 
            patch_count*self.micro_patch_size[0],
            self.micro_patch_size[1], 
            self.c_dim
        ]
        merge_stage1 = generated_patches.reshape(*stage1_shape)
        merge_stage1_slice = []
        for i in range(ratio_over_micro[1]):
            x_st  = self.micro_patch_size[0] * ratio_over_micro[0] * i
            x_ed = x_st + self.micro_patch_size[0] * ratio_over_micro[0]
            y_st  = 0
            y_ed  = self.micro_patch_size[1]
            merge_stage1_slice.append(merge_stage1[:, x_st:x_ed, y_st:y_ed, :])
        merge_stage1_slice = np.concatenate(merge_stage1_slice, axis=2)

        final_shape = [
            -1, 
            ratio_over_micro[0]*self.micro_patch_size[0], 
            ratio_over_micro[1]*self.micro_patch_size[1], 
            self.c_dim
        ]
        merge_stage2 = merge_stage1_slice.reshape(*final_shape)
        return merge_stage2


    def concat_micro_patches_gpu(self, x, ratio_over_micro):

        assert ratio_over_micro[0]==ratio_over_micro[1], "Didn't test x!=y case"
        # ratio_over_micro = int(sqrt(self.full_patch_count))
        num_patches = ratio_over_micro[0] * ratio_over_micro[1]

        # Step 1: micro patches -> stripes of images
        merge_stage1 = tf.reshape(x, [-1, num_patches*self.micro_patch_size[0], self.micro_patch_size[1], 3])
        slices = []
        for i in range(ratio_over_micro[1]):
            slice_st = [0, self.micro_patch_size[0]*ratio_over_micro[0]*i, 0, 0]
            slice_ed = [-1, self.micro_patch_size[1]*ratio_over_micro[0], self.micro_patch_size[1], -1]
            slices.append(tf.slice(merge_stage1, slice_st, slice_ed))
        merge_stage1_slice = tf.concat(slices, axis=2)

        # Step 2: stripes of images -> target image (macro patch or full image)
        final_shape = [
            -1, 
            ratio_over_micro[0]*self.micro_patch_size[0], 
            ratio_over_micro[1]*self.micro_patch_size[1], 
            self.c_dim,
        ]
        merge_stage2 = tf.reshape(merge_stage1_slice, final_shape)
        return merge_stage2

    def crop_micro_from_full_gpu(self, imgs, crop_pos_x, crop_pos_y):

        ps_x, ps_y = self.micro_patch_size # i.e. Patch-Size

        valid_area_x = self.full_image_size[0] - self.micro_patch_size[0]
        if self.coordinate_system == "cylindrical":
            valid_area_y = self.full_image_size[1] # Horizontal don't need padding
        elif self.coordinate_system == "euclidean":
            valid_area_y = self.full_image_size[1] - self.micro_patch_size[1]

        crop_result = []
        batch_size = imgs.shape[0]
        for i in range(batch_size*self.num_micro_compose_macro):
            i_idx = i // self.num_micro_compose_macro
            x_idx = tf.cast(tf.round((crop_pos_x[i, 0]+1)/2*valid_area_x), tf.int32)
            y_idx = tf.cast(tf.round((crop_pos_y[i, 0]+1)/2*valid_area_y), tf.int32)

            # Only cylindrical coordinate system provide overflow protection
            # The code is complicated because:
            #     1. Need to use where to handle "360-degree-edge-crossing" edge case.
            #     2. `tf.where` requires input shape to be the same.
            #
            # P.S. I hate myself selecting TF in the very beginning...
            if self.coordinate_system == "cylindrical":

                # Wrap the end if out-of-bound
                y_idx_st, y_idx_ed = y_idx, y_idx+ps_y
                y_idx_st = tf.where(tf.greater(y_idx_st, self.full_image_size[1]), 
                                    y_idx_st-self.full_image_size[1], 
                                    y_idx_st)
                y_idx_ed = tf.where(tf.greater(y_idx_ed, self.full_image_size[1]), 
                                    y_idx_ed-self.full_image_size[1], 
                                    y_idx_ed)

                # Protect zero selection later, select some trash values instead if the assertion is triggered
                direct_y_idx_st = tf.where(tf.greater(y_idx_st, y_idx_ed), 
                                           y_idx_ed, 
                                           y_idx_st)
                direct_y_idx_ed = tf.where(tf.greater(y_idx_st, y_idx_ed), 
                                           y_idx_st, 
                                           y_idx_ed)

                # `direct_crop` is the default case
                # `wrap_crop` is when the cropped patch will cross the 360 degree line.
                direct_crop = imgs[i_idx, x_idx:x_idx+ps_x, direct_y_idx_st:direct_y_idx_ed, :]
                wrap_crop = tf.concat([
                    imgs[i_idx, x_idx:x_idx+ps_x, y_idx_st:, :],
                    imgs[i_idx, x_idx:x_idx+ps_x, :y_idx_ed, :],
                ], axis=1)

                # Protect selection
                # Remove redundant trash values, force `direct_crop` and `wrap_crop` become the same shape
                direct_crop = direct_crop[:, :ps_y, :]
                wrap_crop   = wrap_crop[:, :ps_y, :]

                selected_crop = tf.where(tf.greater(y_idx_st, y_idx_ed), wrap_crop, direct_crop)
                crop_result.append(selected_crop)
            
            # Euclidean is so easy...
            elif self.coordinate_system == "euclidean":
                y_idx_st = y_idx
                crop_result.append(imgs[i_idx, x_idx:x_idx+ps_x, y_idx:y_idx+ps_y, :])

        return tf.stack(crop_result)

