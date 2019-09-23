import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from fid_utils import fid

BEST_FID_CKPT_DIR = "snapshot_best_fid"
BEST_FID_RECORD_FILENAME = "best_fid_score.txt"

class Evaluator():
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        self.batch_size = config["train_params"]["batch_size"]
        self.fid_step = float(config["log_params"]["fid_step"])
        self.is_calc_fid = (self.fid_step != float("inf"))
        self.num_test_samples = config["data_params"]["num_test_samples"]

        self.input_size = config["data_params"]["full_image_size"]
        self.dataset = config["data_params"]["dataset"].lower()

        if self.is_calc_fid:
            assert self.dataset in {
                "celeba",
                # # we didn't try it
                # "celeba-syn-inward",
                # "celeba-syn-outward",
                # "mp3d",
                "celeba-hq",
                "lsun",
            }, "FID of specified dataset `{}` is not supported!".format(self.dataset)

    def build_graph(self):

        # Build FID graph
        if self.is_calc_fid:
            # load model
            print(" [*] Checking or download inception V3 model..", end=" ", flush=True)
            inception_path = fid.check_or_download_inception(None)
            print("ok")
    
            print(" [*] Loading inception model..", end=" ", flush=True)
            fid.create_inception_graph(inception_path)
            print("ok")

            # load precalculated training set statistics
            print(" [*] Loading pre-calculated FID stats.. ", end="", flush=True)
            fid_stats_path = "./stats/{}_{}x{}_fid_stats.npz".format(self.dataset, self.input_size[0], self.input_size[1])
            if not os.path.exists(fid_stats_path):
                raise ValueError("Can't find pre-calc FID stats, please calc and place it at: \n\t{}".format(fid_stats_path))
            f = np.load(fid_stats_path)
            self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]
            f.close()
            print("ok")
        else:
            print(" [*] FID is disabled!")

    def evaluate(self, trainer):

        if not self.is_calc_fid:
            return 500

        # Extract Inception features
        num_fid_batches = self.num_test_samples//self.batch_size + 1
        all_features = []
        for i in tqdm(range(num_fid_batches)):
            gen_full_images = trainer.rand_sample_full_test()
            gen_full_images = ((gen_full_images + 1.0) * 127.5).astype('uint8')
            batch_features = fid.get_activations(gen_full_images, self.sess, self.batch_size)
            all_features.append(batch_features)
        all_features = np.concatenate(all_features, 0)

        # Calculate FID score, some computers take forever to complete.
        # Please consider change a computer or disable FID calculation
        mu_gen = np.mean(all_features[:self.num_test_samples], axis=0)
        sigma_gen = np.cov(all_features[:self.num_test_samples], rowvar=False)
        try:
            cur_fid = fid.calculate_frechet_distance(mu_gen, sigma_gen, self.mu_real, self.sigma_real)
        except Exception as e:
            print(e)
            cur_fid = 500
        cur_fid = min(cur_fid, 500)
        
        return cur_fid

