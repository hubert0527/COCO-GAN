import os
import glob
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

from img_utils import get_celeba_image, get_image, get_celeba_hq_image
from tqdm import tqdm

import argparse 

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if not os.path.exists('./stats/'):
    os.mkdir('./stats/')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--inception_path', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    n_samples = 50000
    batch_size = 200

    args = parse_args()

    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(args.inception_path) # download inception if necessary
    print("ok")

    # Load image paths and setup
    print("load images..", end=" " , flush=True)
    if 'celeba-hq' in args.dataset.lower():
        if args.resolution==None:
            args.resolution = 1024
        n_samples = 30000
        output_path = './stats/celeba-hq_{}x{}_fid_stats.npz'.format(args.resolution, args.resolution)
        image_list = sorted(glob.glob(args.data_path))[:n_samples]
        image_loader = lambda path: get_celeba_hq_image(path, 1024, 1024, args.resolution, args.resolution)
    elif 'celeba' in args.dataset.lower():
        if args.resolution==None:
            args.resolution = 128
        output_path = './stats/celeba_{}x{}_fid_stats.npz'.format(args.resolution, args.resolution)
        image_list = sorted(glob.glob(args.data_path))[:n_samples]
        image_loader = lambda path: get_celeba_image(path, 256, 256, args.resolution, args.resolution)
    elif 'lsun' in args.dataset.lower():
        if args.resolution==None:
            args.resolution = 256
        output_path = './stats/lsun_{}x{}_fid_stats.npz'.format(args.resolution, args.resolution)
        image_list = sorted(glob.glob(args.data_path))[:n_samples]
        image_loader = lambda path: get_image(path, 256, 256, args.resolution, args.resolution)
    else:
        raise NotImplementedError()
    assert n_samples <= len(image_list)
    print("%d images found and loaded" % len(image_list))


    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")


    print("calculte FID stats..", end=" ", flush=True)
    feature_map = np.zeros((n_samples, 2048))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=64)
        # np.savez_compressed(output_path, mu=mu, sigma=sigma)
        n_batch = n_samples//batch_size
        for batch in tqdm(range(n_batch)):
             images = np.array([image_loader(path) for path in image_list[batch*batch_size:(batch+1)*batch_size]]).astype(np.uint8)
             batch_feature_map = fid.get_activations(images, sess, batch_size, False)
             feature_map[batch*batch_size:(batch+1)*batch_size] = batch_feature_map
        mu = np.mean(feature_map[:n_samples], axis=0)
        sigma = np.cov(feature_map[:n_samples], rowvar=False)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")

