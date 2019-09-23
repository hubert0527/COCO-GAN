import os
import time
import tensorflow as tf
import numpy as np
from numpy import sin, cos


NO_REDUCTION = tf.losses.Reduction.NONE

class Trainer():
    def __init__(self, sess, config, real_images, 
                 g_builder, d_builder, cp_builder, zp_builder, 
                 coord_handler, patch_handler):
        self.sess = sess
        self.config = config
        self.real_images = real_images
        self.g_builder = g_builder
        self.d_builder = d_builder
        self.cp_builder = cp_builder
        self.zp_builder = zp_builder
        self.coord_handler = coord_handler
        self.patch_handler = patch_handler

        # Vars for graph building
        self.batch_size = self.config["train_params"]["batch_size"]
        self.z_dim = self.config["model_params"]["z_dim"]
        self.spatial_dim = self.config["model_params"]["spatial_dim"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]

        self.ratio_macro_to_micro = self.config["data_params"]["ratio_macro_to_micro"]
        self.ratio_full_to_micro = self.config["data_params"]["ratio_full_to_micro"]
        self.num_micro_compose_macro = self.config["data_params"]["num_micro_compose_macro"]

        # Vars for training loop
        self.exp_name = config["log_params"]["exp_name"]
        self.epochs = float(self.config["train_params"]["epochs"])
        self.num_batches = self.config["data_params"]["num_train_samples"] // self.batch_size
        self.coordinate_system = self.config["data_params"]["coordinate_system"]
        self.G_update_period = self.config["train_params"]["G_update_period"]
        self.D_update_period = self.config["train_params"]["D_update_period"]
        self.Q_update_period = self.config["train_params"]["Q_update_period"]

        # Loss weights
        self.code_loss_w = self.config["loss_params"]["code_loss_w"]
        self.coord_loss_w = self.config["loss_params"]["coord_loss_w"]
        self.gp_lambda = self.config["loss_params"]["gp_lambda"]

        # Extrapolation parameters handling
        self.train_extrap = self.config["train_params"]["train_extrap"]
        if self.train_extrap:
            assert self.config["train_params"]["num_extrap_steps"] is not None
            assert self.coordinate_system is not "euclidean", \
                "I didn't handle extrapolation in {} coordinate system!".format(self.coordinate_system)
            self.num_extrap_steps = self.config["train_params"]["num_extrap_steps"]
        else:
            self.num_extrap_steps = 0


    def _train_content_prediction_model(self):
        return (self.Q_update_period>0) and (self.config["train_params"]["qlr"]>0)


    def sample_prior(self):
        return np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)

    
    def _dup_z_for_macro(self, z):
        # Duplicate with nearest neighbor, different to `tf.tile`.
        # E.g., 
        # tensor: [[1, 2], [3, 4]]
        # repeat: 3
        # output: [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
        ch = z.shape[-1]
        repeat = self.num_micro_compose_macro
        extend = tf.expand_dims(z, 1)
        extend_dup = tf.tile(extend, [1, repeat, 1])
        return tf.reshape(extend_dup, [-1, ch])


    def build_graph(self):

        # Input nodes
        # Note: the input node name was wrong in the checkpoint 
        self.micro_coord_fake = tf.placeholder(tf.float32, [None, self.spatial_dim], name='micro_coord_fake')
        self.macro_coord_fake = tf.placeholder(tf.float32, [None, self.spatial_dim], name='macro_coord_fake')
        self.micro_coord_real = tf.placeholder(tf.float32, [None, self.spatial_dim], name='micro_coord_real')
        self.macro_coord_real = tf.placeholder(tf.float32, [None, self.spatial_dim], name='macro_coord_real')

        # Reversing angle for cylindrical coordinate is complicated, directly pass values here
        self.y_angle_ratio = tf.placeholder(tf.float32, [None, 1], name='y_angle_ratio') 
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        
        # Crop real micro for visualization
        if self.coordinate_system == "euclidean":
             self.real_micro = self.patch_handler.crop_micro_from_full_gpu(
                self.real_images, self.micro_coord_real[:, 0:1], self.micro_coord_real[:, 1:2])
        elif self.coordinate_system == "cylindrical":
            self.real_micro = self.patch_handler.crop_micro_from_full_gpu(
                self.real_images, self.micro_coord_real[:, 0:1], self.y_angle_ratio)

        # Real part
        self.real_macro = self.patch_handler.concat_micro_patches_gpu(
            self.real_micro, ratio_over_micro=self.ratio_macro_to_micro)
        (self.disc_real, disc_real_h) = self.d_builder(self.real_macro, self.macro_coord_real, is_training=True)
        self.c_real_pred = self.cp_builder(disc_real_h, is_training=True)
        self.z_real_pred = self.zp_builder(disc_real_h, is_training=True)

        # Fake part
        z_dup_macro = self._dup_z_for_macro(self.z)
        self.gen_micro = self.g_builder(z_dup_macro, self.micro_coord_fake, is_training=True)
        self.gen_macro = self.patch_handler.concat_micro_patches_gpu(
            self.gen_micro, ratio_over_micro=self.ratio_macro_to_micro)
        (self.disc_fake, disc_fake_h) = self.d_builder(self.gen_macro, self.macro_coord_fake, is_training=True)
        self.c_fake_pred = self.cp_builder(disc_fake_h, is_training=True)
        self.z_fake_pred = self.zp_builder(disc_fake_h, is_training=True)

        # Testing graph
        if self.config["log_params"]["merge_micro_patches_in_cpu"]:
            self.gen_micro_test = self.g_builder(self.z, self.micro_coord_fake, is_training=False)
        else:
            (self.gen_micro_test, self.gen_full_test) = self.generate_full_image_gpu(self.z)

        # Patch-Guided Image Generation graph
        if self._train_content_prediction_model():
            (_, disc_real_h_rec) = self.d_builder(self.real_macro, None, is_training=False)
            estim_z = self.zp_builder(disc_real_h_rec, is_training=False)
            # I didn't especially handle this.
            # if self.config["log_params"]["merge_micro_patches_in_cpu"]:
            (_, self.rec_full) = self.generate_full_image_gpu(self.z)

        print(" [Build] Composing Loss Functions ")
        self._compose_losses()

        print(" [Build] Creating Optimizers ")
        self._create_optimizers()


    def _calc_gradient_penalty(self):
        """ Gradient Penalty for patches D """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=tf.shape(self.real_macro), minval=0.,maxval=1.)
        differences = self.gen_macro - self.real_macro # This is different from MAGAN
        interpolates = self.real_macro + (alpha * differences)
        disc_inter, _ = self.d_builder(interpolates, None, is_training=True)
        gradients = tf.gradients(disc_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        return gradient_penalty, slopes


    def _compose_losses(self):

        # Content consistency loss
        self.code_loss = tf.reduce_mean(self.code_loss_w * tf.losses.absolute_difference(self.z, self.z_fake_pred))

        # Spatial consistency loss (reduce later)
        self.coord_mse_real = self.coord_loss_w * tf.losses.mean_squared_error(self.macro_coord_real, self.c_real_pred, reduction=NO_REDUCTION)
        self.coord_mse_fake = self.coord_loss_w * tf.losses.mean_squared_error(self.macro_coord_fake, self.c_fake_pred, reduction=NO_REDUCTION)

        # (For extrapolation training) Mask-out out-of-bound (OOB) coordinate loss since the gradients are useless
        if self.train_extrap:
            upper_bound = tf.ones([self.batch_size, self.spatial_dim], tf.float32) + 1e-4
            lower_bound = - upper_bound
            exceed_upper_bound = tf.greater(self.macro_coord_fake, upper_bound)
            exceed_lower_bound = tf.less(self.macro_coord_fake, lower_bound)

            oob_mask_sep   = tf.math.logical_or(exceed_upper_bound, exceed_lower_bound)
            oob_mask_merge = tf.math.logical_or(oob_mask_sep[:, 0], oob_mask_sep[:, 1])
            for i in range(2, self.spatial_dim):
                oob_mask_merge = tf.math.logical_or(oob_mask_merge, oob_mask_sep[:, i])
            oob_mask = tf.tile(tf.expand_dims(oob_mask_merge, 1), [1, self.spatial_dim])
            self.coord_mse_fake = tf.where(oob_mask, tf.stop_gradient(self.coord_mse_fake), self.coord_mse_fake)

        self.coord_mse_real = tf.reduce_mean(self.coord_mse_real)
        self.coord_mse_fake = tf.reduce_mean(self.coord_mse_fake)
        self.coord_loss = self.coord_mse_real + self.coord_mse_fake

        # WGAN loss
        self.adv_real = - tf.reduce_mean(self.disc_real)
        self.adv_fake = tf.reduce_mean(self.disc_fake)
        self.d_adv_loss = self.adv_real + self.adv_fake
        self.g_adv_loss = - self.adv_fake

        # Gradient penalty loss of WGAN-GP
        gradient_penalty, self.gp_slopes = self._calc_gradient_penalty()
        self.gp_loss = self.config["loss_params"]["gp_lambda"] * gradient_penalty

        # Total loss
        self.d_loss = self.d_adv_loss + self.gp_loss + self.coord_loss + self.code_loss
        self.g_loss = self.g_adv_loss + self.coord_loss + self.code_loss
        self.q_loss = self.g_adv_loss + self.code_loss

        # Wasserstein distance for visualization
        self.w_dist = - self.adv_real - self.adv_fake

        
    def _create_optimizers(self):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'D' in var.name]
        g_vars = [var for var in t_vars if 'G' in var.name]
        q_vars = [var for var in t_vars if 'Q' in var.name]
        
        # optimizers
        G_update_ops = tf.get_collection(self.g_builder.update_collection)
        D_update_ops = tf.get_collection(self.d_builder.update_collection)
        Q_update_ops = tf.get_collection(self.zp_builder.update_collection)
        GD_update_ops = tf.get_collection(self.cp_builder.update_collection)

        with tf.control_dependencies(G_update_ops + GD_update_ops):
            self.g_optim = tf.train.AdamOptimizer(
                self.config["train_params"]["glr"], 
                beta1=self.config["train_params"]["beta1"], 
                beta2=self.config["train_params"]["beta2"], 
            ).minimize(self.g_loss, var_list=g_vars)

        with tf.control_dependencies(D_update_ops + GD_update_ops):
            self.d_optim = tf.train.AdamOptimizer(
                self.config["train_params"]["dlr"],
                beta1=self.config["train_params"]["beta1"], 
                beta2=self.config["train_params"]["beta2"], 
            ).minimize(self.d_loss, var_list=d_vars)

        if self._train_content_prediction_model():
            with tf.control_dependencies(Q_update_ops):
                self.q_optim = tf.train.AdamOptimizer(
                    self.config["train_params"]["qlr"],
                    beta1=self.config["train_params"]["beta1"], 
                    beta2=self.config["train_params"]["beta2"], 
                ).minimize(self.q_loss, var_list=q_vars)

        if self.train_extrap:
            with tf.variable_scope("extrap_optim"):
                g_vars_partial = [
                    var for var in g_vars if ("g_resblock_0" in var.name or "g_resblock_1" in var.name)] 
                with tf.control_dependencies(G_update_ops + GD_update_ops):
                    self.g_optim_extrap = tf.train.AdamOptimizer(
                        self.config["train_params"]["glr"], 
                        beta1=self.config["train_params"]["beta1"], 
                        beta2=self.config["train_params"]["beta2"], 
                    ).minimize(self.g_loss, var_list=g_vars_partial)
    
                with tf.control_dependencies(D_update_ops + GD_update_ops):
                    self.d_optim_extrap = tf.train.AdamOptimizer(
                        self.config["train_params"]["dlr"], 
                        beta1=self.config["train_params"]["beta1"], 
                        beta2=self.config["train_params"]["beta2"], 
                    ).minimize(self.d_loss, var_list=d_vars)


    def rand_sample_full_test(self):
        if self.config["log_params"]["merge_micro_patches_in_cpu"]:
            z = self.sample_prior()
            _, full_images = self.generate_full_image_cpu(z)
        else:
            full_images = self.sess.run(
                self.gen_full_test, feed_dict={self.z: self.sample_prior()})
        return full_images

    
    def generate_full_image_gpu(self, z):
        all_micro_patches = []
        all_micro_coord = []
        num_patches_x = self.ratio_full_to_micro[0] + self.num_extrap_steps*2
        num_patches_y = self.ratio_full_to_micro[1] + self.num_extrap_steps*2
        for yy in range(num_patches_y):
            for xx in range(num_patches_x):
                if self.coordinate_system == "euclidean":
                    micro_coord_single = tf.constant([
                        self.coord_handler.euclidean_coord_int_full_to_float_micro(xx, num_patches_x, extrap_steps=self.num_extrap_steps), 
                        self.coord_handler.euclidean_coord_int_full_to_float_micro(yy, num_patches_y, extrap_steps=self.num_extrap_steps),
                    ])
                elif self.coordinate_system == "cylindrical":
                    theta_ratio = self.coord_handler.hyperbolic_coord_int_full_to_float_micro(yy, num_patches_y)
                    micro_coord_single = tf.constant([
                        self.coord_handler.euclidean_coord_int_full_to_float_micro(xx, num_patches_x), 
                        self.coord_handler.hyperbolic_theta_to_euclidean(theta_ratio, proj_func=cos),
                        self.coord_handler.hyperbolic_theta_to_euclidean(theta_ratio, proj_func=sin),
                    ])
                micro_coord = tf.tile(tf.expand_dims(micro_coord_single, 0), [tf.shape(z)[0], 1])
                generated_patch = self.g_builder(z, micro_coord, is_training=False)
                all_micro_patches.append(generated_patch)
                all_micro_coord.append(micro_coord)

        num_patches = num_patches_x * num_patches_y
        all_micro_patches = tf.concat(all_micro_patches, 0)
        all_micro_patches_reord = self.patch_handler.reord_patches_gpu(all_micro_patches, self.batch_size, num_patches)
        full_image = self.patch_handler.concat_micro_patches_gpu(
            all_micro_patches_reord, 
            ratio_over_micro=[num_patches_x, num_patches_y])

        return all_micro_patches, full_image


    def generate_full_image_cpu(self, z):
        all_micro_patches = []
        all_micro_coord = []
        num_patches_x = self.ratio_full_to_micro[0] + self.num_extrap_steps * 2
        num_patches_y = self.ratio_full_to_micro[1] + self.num_extrap_steps * 2
        for yy in range(num_patches_y):
            for xx in range(num_patches_x):
                if self.coordinate_system == "euclidean":
                    micro_coord_single = np.array([
                        self.coord_handler.euclidean_coord_int_full_to_float_micro(xx, num_patches_x, extrap_steps=self.num_extrap_steps),
                        self.coord_handler.euclidean_coord_int_full_to_float_micro(yy, num_patches_y, extrap_steps=self.num_extrap_steps),
                    ])
                elif self.coordinate_system == "cylindrical":
                    theta_ratio = self.coord_handler.hyperbolic_coord_int_full_to_float_micro(yy, num_patches_y)
                    micro_coord_single = np.array([
                        self.coord_handler.euclidean_coord_int_full_to_float_micro(xx, num_patches_x),
                        self.coord_handler.hyperbolic_theta_to_euclidean(theta_ratio, proj_func=cos),
                        self.coord_handler.hyperbolic_theta_to_euclidean(theta_ratio, proj_func=sin),
                    ])
                micro_coord = np.tile(np.expand_dims(micro_coord_single, 0), [z.shape[0], 1])
                generated_patch = self.sess.run(
                    self.gen_micro_test, feed_dict={self.z: z, self.micro_coord_fake: micro_coord}) # TODO
                all_micro_patches.append(generated_patch)
                all_micro_coord.append(micro_coord)

        num_patches = num_patches_x * num_patches_y
        all_micro_patches = np.concatenate(all_micro_patches, 0)
        all_micro_patches_reord = self.patch_handler.reord_patches_cpu(all_micro_patches, self.batch_size, num_patches)
        full_image = self.patch_handler.concat_micro_patches_cpu(
            all_micro_patches_reord, 
            ratio_over_micro=[num_patches_x, num_patches_y])

        return all_micro_patches, full_image


    def train(self, logger, evaluator, global_step):
        start_time = time.time()
        g_loss, d_loss, q_loss = 0, 0, 0
        z_fixed = self.sample_prior()
        cur_epoch = int(global_step / self.num_batches)
        cur_iter  = global_step - cur_epoch * self.num_batches

        while cur_epoch < self.epochs:
            while cur_iter < self.num_batches:

                # Create data
                z_iter = self.sample_prior()
                macro_coord, micro_coord, y_angle_ratio = self.coord_handler.sample_coord()
                feed_dict_iter = {
                    self.micro_coord_real: micro_coord,
                    self.macro_coord_real: macro_coord,
                    self.micro_coord_fake: micro_coord,
                    self.macro_coord_fake: macro_coord,
                    self.y_angle_ratio: y_angle_ratio,
                    self.z: z_iter,
                }
                feed_dict_fixed = {
                    self.micro_coord_real: micro_coord,
                    self.macro_coord_real: macro_coord,
                    self.micro_coord_fake: micro_coord,
                    self.macro_coord_fake: macro_coord,
                    self.y_angle_ratio: y_angle_ratio,
                    self.z: z_fixed,
                }
                
                # Optimize
                if (global_step % self.D_update_period) == 0:
                    _, d_summary_str, d_loss = self.sess.run(
                        [self.d_optim, logger.d_summaries, self.d_loss], 
                        feed_dict=feed_dict_iter)
                if (global_step % self.G_update_period) == 0:
                    _, g_summary_str, g_loss = self.sess.run(
                        [self.g_optim, logger.g_summaries, self.g_loss], 
                        feed_dict=feed_dict_iter)

                if self.train_extrap:
                    macro_coord_extrap, micro_coord_extrap, _ = \
                        self.coord_handler.sample_coord(num_extrap_steps=self.num_extrap_steps)
                    # Override logging inputs as well
                    feed_dict_fixed[self.micro_coord_fake] = micro_coord_extrap
                    feed_dict_fixed[self.macro_coord_fake] = macro_coord_extrap
                    feed_dict_iter[self.micro_coord_fake] = micro_coord_extrap
                    feed_dict_iter[self.macro_coord_fake] = macro_coord_extrap

                    if (global_step % self.D_update_period) == 0:
                        _, d_summary_str, d_loss = self.sess.run(
                             [self.d_optim_extrap, logger.d_summaries, self.d_loss], 
                             feed_dict=feed_dict_iter)
                    if (global_step % self.G_update_period) == 0:
                        _, g_summary_str, g_loss = self.sess.run(
                            [self.g_optim_extrap, logger.g_summaries, self.g_loss], 
                            feed_dict=feed_dict_iter)

                if self._train_content_prediction_model() and (global_step % self.Q_update_period) == 0:
                    _, q_loss = self.sess.run(
                        [self.q_optim, self.q_loss], 
                        feed_dict=feed_dict_iter)

                # Log
                time_elapsed = time.time() - start_time
                print("[{}] [Epoch: {}; {:4d}/{:4d}; global_step:{}] elapsed: {:.4f}, d: {:.4f}, g: {:.4f}, q: {:.4f}".format(
                    self.exp_name, cur_epoch, cur_iter, self.num_batches, global_step, time_elapsed, d_loss, g_loss, q_loss))
                logger.log_iter(self, evaluator, cur_epoch, cur_iter, global_step, g_summary_str, d_summary_str, 
                                z_iter, z_fixed, feed_dict_iter, feed_dict_fixed)

                cur_iter += 1
                global_step += 1
                
            cur_epoch += 1
            cur_iter = 0
