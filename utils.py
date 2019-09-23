import os
import tensorflow as tf
import numpy as np
from scipy.misc import imsave


def aug_cylindrical_data_tensor(batch_images_t):
    width = batch_images_t.shape[2].value
    rotate_dist = tf.round(tf.random.uniform([], 0, 1) * width)
    rotate_dist = tf.cast(rotate_dist, tf.int32)
    batch_aug_results = tf.concat([
        batch_images_t[:, :, rotate_dist:], batch_images_t[:, :, :rotate_dist]
    ], axis=2)
    return batch_aug_results


def aug_cylindrical_data_numpy(batch_images):
    width = batch_images_t.shape[2]
    rotate_dist = int(round(np.random.uniform(0, 1) * width))
    batch_aug_results = np.concatenate([
        batch_images[:, :, rotate_dist:], batch_images[:, :, :rotate_dist]
    ], axis=2)
    return batch_aug_results


def save_manifold_images(images, size, image_path):
    images = (images+1) / 2
    manifold_image = np.squeeze(compose_manifold_images(images, size))
    return imsave(image_path, manifold_image)


def compose_manifold_images(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ' + 
            'must have dimensions: HxW or HxWx3 or HxWx4, got {}'.format(images.shape))
