from glob import glob
import os
import random
import scipy.misc as misc
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--syn_type", type=str, choices=["inward", "outward"])
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

configs = {
	"inward": {
		"x": 25,
		"y": 50,
		"resolution": 128,
		"delta_x": [-25, 25],
		"delta_y": [-25, 25],
		"out_dir": "./data/CelebA_syn_inward/",
	},
	"outward": {
		"x": 12,
		"y": 25,
		"resolution": 150,
		"delta_x": [12, 25],
		"delta_y": [-12, 16],
		"out_dir": "./data/CelebA_syn_outward/",
	}
}

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def run_aug(resolution, x, y, delta_x, delta_y, out_dir='argu_imgs'):
	out_dir = check_folder(out_dir)
	for data_name in glob(args.data_dir):
		data = misc.imread(data_name)
		rand_y = y + random.randint(delta_y[0], delta_y[1])
		rand_x = x + random.randint(delta_x[0], delta_x[1])
		save_img = data[rand_y:rand_y+resolution, rand_x:rand_x+resolution, :]
		save_name = data_name.split('/')[-1]
		misc.imsave(os.path.join(out_dir, save_name), save_img)

config = configs[args.syn_type]
run_aug(config["resolution"], config["x"], config["y"], config["delta_x"], config["delta_y"], out_dir=config["out_dir"])

