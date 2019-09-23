import os
import tensorflow as tf
import numpy as np
import re

from utils import save_manifold_images

BEST_FID_CKPT_DIR = "snapshot_best_fid"
BEST_FID_RECORD_FILENAME = "best_fid_score.txt"

class Logger():
    def __init__(self, sess, config, patch_handler):
        self.sess = sess
        self.config = config
        self.patch_handler = patch_handler
        self.best_fid = 500

        self.batch_size = self.config["train_params"]["batch_size"]
        self.log_full_with_cpu = self.config["log_params"]["merge_micro_patches_in_cpu"]
        self.num_micro_compose_full = self.config["data_params"]["num_micro_compose_full"]

        if self.config["train_params"]["train_extrap"]:
            num_extrap_steps = self.config["train_params"]["num_extrap_steps"]
            micro_patch_size = self.config["data_params"]["micro_patch_size"]
            extrap_size_x = num_extrap_steps * micro_patch_size[0] * 2
            extrap_size_y = num_extrap_steps * micro_patch_size[1] * 2
            self.full_shape = [
                None, 
                self.config["data_params"]["full_image_size"][0] + extrap_size_x, 
                self.config["data_params"]["full_image_size"][1] + extrap_size_y, 
                self.config["data_params"]["c_dim"], 
            ]
        else:
            self.full_shape = [
                None, 
                self.config["data_params"]["full_image_size"][0], 
                self.config["data_params"]["full_image_size"][1], 
                self.config["data_params"]["c_dim"], 
            ]


        self.exp_name = config["log_params"]["exp_name"]
        self.log_dir = self._check_folder(os.path.join(config["log_params"]["log_dir"], self.exp_name))
        self.ckpt_dir = self._check_folder(os.path.join(self.log_dir, "ckpt"))
        self.img_dir = self._check_folder(os.path.join(self.log_dir, "images"))
        self.force_load_from_dir = self.config["train_params"]["force_load_from_dir"]

        # Use float to parse "inf"
        self.log_step = float(config["log_params"]["log_step"])
        self.img_step = float(config["log_params"]["img_step"])
        self.fid_step = float(config["log_params"]["fid_step"])
        self.ckpt_step = float(config["log_params"]["ckpt_step"])
        self.dump_img_step = float(config["log_params"]["dump_img_step"])

        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)


    def _check_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder


    def _build_numerical_summaries(self, trainer):

        self.fid_tfvar = tf.Variable(0.0, trainable=False)
        self.fid_summary = tf.summary.scalar("FID/FID", self.fid_tfvar)

        self.g_summaries = tf.summary.merge([
            tf.summary.scalar("total_loss/g_loss", trainer.g_loss),
        ])

        self.d_summaries = tf.summary.merge([

            # Main losses
            tf.summary.scalar("total_loss/d_loss", trainer.d_loss), 

            tf.summary.scalar("D/gp_loss", trainer.gp_loss),
            tf.summary.scalar("D/adv_real", trainer.adv_real), 
            tf.summary.scalar("D/adv_fake", trainer.adv_fake),
            tf.summary.scalar("code/code_fake_loss", trainer.code_loss),
            tf.summary.scalar("coord/coord_mse_real", trainer.coord_mse_real), 
            tf.summary.scalar("coord/coord_mse_fake", trainer.coord_mse_fake),

            # Monitoring training quality
            tf.summary.scalar("w_dist", trainer.w_dist),
            tf.summary.histogram("gp_slopes", trainer.gp_slopes),
            tf.summary.histogram('code/z_real_pred', trainer.z_real_pred),
            tf.summary.histogram('code/z_fake_pred', trainer.z_fake_pred), 
            tf.summary.histogram('coord/c_real_pred', trainer.c_real_pred),
            tf.summary.histogram('coord/c_fake_pred', trainer.c_fake_pred),

            # Debugging 
            tf.summary.histogram('input/micro_coord_real', trainer.micro_coord_real), 
            tf.summary.histogram('input/macro_coord_real', trainer.macro_coord_real),
            tf.summary.histogram('input/micro_coord_fake', trainer.micro_coord_fake), 
            tf.summary.histogram('input/macro_coord_fake', trainer.macro_coord_fake),
        ])


    def _build_img_summaries(self, trainer):
        img_summaries = [
            tf.summary.image('fake_micro', trainer.gen_micro, max_outputs=3),
            tf.summary.image('fake_macro', trainer.gen_macro, max_outputs=3),
            tf.summary.image('real_micro', trainer.real_micro, max_outputs=3),
            tf.summary.image('real_macro', trainer.real_macro, max_outputs=3),
        ]

        if self.log_full_with_cpu:
            self.gen_full_sum_input = \
                tf.placeholder(tf.float32, self.full_shape, name='gen_full_sum_input')
            self.fake_full_summary = tf.summary.image('fake_full', self.gen_full_sum_input, max_outputs=3)
        else:
            img_summaries.append(tf.summary.image('fake_full', trainer.gen_full_test, max_outputs=3))

        if trainer._train_content_prediction_model():
            img_summaries.append(tf.summary.image('patch_guided_gen', trainer.rec_full, max_outputs=3))
                
        self.img_summaries = tf.summary.merge(img_summaries)


    def build_graph(self, trainer):
        self._build_numerical_summaries(trainer)
        self._build_img_summaries(trainer)

        if trainer.train_extrap:
            # Extrap training has an additional Adam optimizer with parameters not existed in the ckpt
            ckpt_dir = self.force_load_from_dir if self.force_load_from_dir else self.ckpt_dir
            ckpt_vars = set([v[0] for v in tf.train.list_variables(ckpt_dir)])
            restore_var = [v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if v.op.name in ckpt_vars]
            self.saver = tf.train.Saver(max_to_keep=3, var_list=restore_var)
        else:
            self.saver = tf.train.Saver(max_to_keep=3)


    def _check_step(self, step, step_config):
        if step_config is None:
            return False
        elif step==0:
            return False
        return (step % step_config) == 0


    def log_iter(self, trainer, evaluator, epoch, iter_, global_step, g_summary_str, d_summary_str, 
                 z_iter, z_fixed, feed_dict_iter, feed_dict_fixed):

        # Write numerical stats
        if self._check_step(global_step, self.log_step):
            self.writer.add_summary(g_summary_str, global_step)
            self.writer.add_summary(d_summary_str, global_step)

        # Write patches/images to TensorBoard
        if self._check_step(global_step, self.img_step):
            img_summary_str = self.sess.run(self.img_summaries, feed_dict=feed_dict_iter)
            self.writer.add_summary(img_summary_str, global_step)
            
            if self.log_full_with_cpu:
                _, full_images = trainer.generate_full_image_cpu(z_iter)
                full_img_summary_str = self.sess.run(
                    self.fake_full_summary, feed_dict={self.gen_full_sum_input: full_images})
                self.writer.add_summary(full_img_summary_str, global_step)
            else:
                # The full image summary in GPU mdoe is directly combined with other image summaries
                pass
            
        # Save results to disk
        # We use a set of fixed z here to better monitor the changes through time.
        if self._check_step(global_step, self.dump_img_step):

            if self.log_full_with_cpu:
                fixed_patch, fixed_full = trainer.generate_full_image_cpu(z_fixed)
                _, sampled_full = trainer.generate_full_image_cpu(z_iter)
            else:
                fixed_patch, fixed_full = \
                    self.sess.run([trainer.gen_micro_test, trainer.gen_full_test], feed_dict=feed_dict_fixed)
                _, sampled_full = \
                    self.sess.run([trainer.gen_micro_test, trainer.gen_full_test], feed_dict=feed_dict_iter)

            num_full = self.batch_size
            num_patches = self.batch_size * self.num_micro_compose_full
            manifold_h_f, manifold_w_f = int(np.sqrt(num_full)), int(np.sqrt(num_full))
            manifold_h_p, manifold_w_p = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
            
            # Save fixed micro patches
            save_name = 'fixed_patch_{:02d}_{:04d}.png'.format(epoch, iter_)
            save_manifold_images(fixed_patch[:manifold_h_p * manifold_w_p, :, :, :], 
                                 [manifold_h_p, manifold_w_p], 
                                 os.path.join(self.img_dir, save_name))
            
            # Save fixed full images
            save_name = 'fixed_full_{:02d}_{:04d}.png'.format(epoch, iter_)
            save_manifold_images(fixed_full[:manifold_h_f * manifold_w_f, :, :, :], 
                                 [manifold_h_f, manifold_w_f], 
                                 os.path.join(self.img_dir, save_name))

            # Save sampled full images
            save_name = 'sampled_full_{:02d}_{:04d}.png'.format(epoch, iter_)
            save_manifold_images(sampled_full[:manifold_h_f * manifold_w_f, :, :, :], 
                                 [manifold_h_f, manifold_w_f], 
                                 os.path.join(self.img_dir, save_name))

        # Calc FID
        if self._check_step(global_step, self.fid_step):
            cur_fid = evaluator.evaluate(trainer)
            self.sess.run(tf.assign(self.fid_tfvar, cur_fid))
            fid_summary_str = self.sess.run(self.fid_summary)
            self.writer.add_summary(fid_summary_str, global_step)
            if cur_fid < self.best_fid:
                self.best_fid = cur_fid
                self.save(global_step, extra_dir=BEST_FID_CKPT_DIR)

        # Save model
        if self._check_step(global_step, self.ckpt_step):
            self.save(global_step)


    def save(self, global_step, extra_dir=None):
        # Standard saving
        if extra_dir is None:
            target_dir = self.ckpt_dir
            self.saver.save(self.sess, os.path.join(target_dir, 'model.ckpt'), global_step=global_step)
        # Save to a new target, needs to create new saver
        else:
            target_dir = os.path.join(self.ckpt_dir, extra_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            tf.train.Saver(max_to_keep=3).save(self.sess, os.path.join(target_dir, 'model.ckpt'), global_step=global_step)

        if self.fid_step != float("inf"):
            self.dump_best_fid(target_dir)


    def load_ckpt(self):
        if self.force_load_from_dir:
            from_dir = self.force_load_from_dir
        else:
            from_dir = self.ckpt_dir
            
        print(" [*] Reading checkpoint from `{}`...".format(from_dir))
        load_success = False
        ckpt = tf.train.get_checkpoint_state(from_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_filename = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(from_dir, ckpt_filename))
            global_step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_filename)).group(0))
            print(" [*] Success to read {}".format(ckpt_filename))
            try:
                self.best_fid = self.load_best_fid(from_dir)
                load_success = True
                print(" [*] Success to load best FID score. Start from best_fid={}".format(self.best_fid))
            except Exception as e:
                self.best_fid = 500
                load_success = False
                print(" [*] Failed to load best FID score. Start from best_fid={}".format(500), e)
        else:
            print(" [*] Failed to find a checkpoint")
            load_success = False
            global_step = 0

        if self.config["train_params"]["train_extrap"]:
            assert load_success and self.force_load_from_dir not in {False, None, ""}, \
                "Post-training extrapolation must load from pretrained model! You loaded from: {}".format(self.force_load_from_dir)

        return global_step

    
    def dump_best_fid(self, save_dir):
        with open(os.path.join(save_dir, BEST_FID_RECORD_FILENAME), 'w') as f:
            f.write(str(self.best_fid))


    def load_best_fid(self, save_dir):
        path = os.path.join(save_dir, BEST_FID_RECORD_FILENAME)
        if not os.path.exists(path):
            return 500
        with open(path, 'r') as f:
            best_fid = float(f.readline())
        return best_fid
