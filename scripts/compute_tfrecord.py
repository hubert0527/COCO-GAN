import os 
import sys
sys.path.append(".") # WTF

import tensorflow as tf 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
from glob import glob
import threading 
import argparse

from img_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["celeba", "lsun", "celeba-hq", "matterport3d"])
parser.add_argument('--img_paths', type=str, default=None)
parser.add_argument('--resolution', type=int, default=None)
args = parser.parse_args()

root_path = "./data/"

"""Constants for dataset setup"""
res_x, res_y = args.resolution, args.resolution
if args.dataset.lower() in {'celeba'}:
    raw_x, raw_y = 256, 256
    if args.resolution is None:
        res_x, res_y = 128, 128
    img_fn = lambda img_path: get_celeba_image(img_path, raw_x, raw_y, res_x, res_y)
    img_paths = os.path.join(root_path, 'CelebA/*.jpg')
elif args.dataset.lower()=='lsun':
    raw_x, raw_y = 256, 256
    if args.resolution is None:
        res_x, res_y = raw_x, raw_y
    img_fn = lambda img_path: get_image(img_path, raw_x, raw_y, res_x, res_y)
    img_paths = os.path.join(root_path, 'lsun/*.webp')
elif args.dataset.lower()=='celeba-hq':
    raw_x, raw_y = 1024, 1024
    if args.resolution is None:
        res_x, res_y = raw_x, raw_y
    img_fn = lambda img_path: get_celeba_hq_image(img_path, raw_x, raw_y, res_x, res_y)
    img_paths = os.path.join(root_path, 'CelebA-HQ/*.npy')
elif args.dataset.lower()=='matterport3d':
    assert args.resolution == None
    raw_x, raw_y = 256, 768
    res_x, res_y = raw_x, raw_y
    img_fn = lambda img_path: get_image(img_path, 256, 768, res_x, res_y)
    img_paths = os.path.join(root_path, 'matterport3d_panorama/*.png')
elif args.dataset.lower()=='CelebA_syn_inward'.lower():
    assert args.resolution == None
    res_x, res_y = 128, 128
    img_fn = lambda img_path: get_image(img_path, 128, 128, res_x, res_x)
    img_paths = os.path.join(root_path, 'CelebA_syn_inward/*.png')
elif args.dataset.lower()=='CelebA_syn_inward'.lower():
    assert args.resolution == None
    res_x, res_y = 128, 128
    img_fn = lambda img_path: get_image(img_path, 150, 150, res_x, res_x)
    img_paths = os.path.join(root_path, 'CelebA_syn_outward/*.png')
else:
    raise NotImplementedError("Dataset: {}, is not implemented".format(args.dataset))


class DataComposer(threading.Thread):  
    def  __init__( self, lock, img_fn, batch_paths, writer, tid):
        super(DataComposer, self).__init__(name="thread_{}".format(tid))
        self.lock = lock  
        self.batch_paths = batch_paths
        self.writer = writer
        self.img_fn = img_fn

    def _process_image_files_batch(self):
        for img_path in self.batch_paths:
            img = self.img_fn(img_path).tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            self.lock.acquire()
            self.writer.write(example.SerializeToString())
            self.lock.release()

    def run(self):
        self._process_image_files_batch()


if __name__ == "__main__":
    if args.img_paths is not None:
        img_paths = args.img_paths
    print(" [*] Reading all paths from {}".format(img_paths))
    all_paths   = glob(img_paths)
    assert len(all_paths)>0, "Can't find any image at {}".format(img_paths)

    n_process   = 36
    batch_size  = 100
    steps_per_spawn = batch_size * n_process
    num_batches     = len(all_paths) // batch_size + 1
    n_steps         = len(all_paths) // steps_per_spawn + 1

    print(" [*] Generating exp: {}!".format("{}_{}x{}.tfrecords".format(args.dataset, res_x, res_y)))
    print(" [*] {} image paths loaded, {} steps to spawn!".format(len(all_paths), n_steps))

    if not os.path.exists("./tfrecords"):
        os.makedirs("./tfrecords")
    coord = tf.train.Coordinator()
    semaphore = threading.BoundedSemaphore(1)
    writer= tf.python_io.TFRecordWriter("./tfrecords/{}_{}x{}.tfrecords".format(args.dataset, res_x, res_y))

    for i in tqdm(range(n_steps)):
        global_steps = i * steps_per_spawn
        threads = []
        for p in range(n_process):
            st = global_steps + p * batch_size
            ed = st + batch_size
            paths_list = all_paths[st:ed]
            t = DataComposer(semaphore, img_fn, paths_list, writer, tid=p)
            t.start()
            threads.append(t)
        coord.join(threads)
    writer.close() 
