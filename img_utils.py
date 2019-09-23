import scipy.misc
import numpy as np

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, byte_image=True):
    image = imread(image_path)
    return transform(image, input_height, input_width, resize_height, resize_width, byte_image=byte_image)

def get_celeba_hq_image(image_path, input_height, input_width, resize_height=64, resize_width=64, byte_image=True):
    image = np.load(image_path).astype(np.float32)
    return transform(image, input_height, input_width, resize_height, resize_width, byte_image=byte_image)

def get_celeba_image(image_path, input_height, input_width, resize_height=64, resize_width=64, byte_image=True):
    image = imread(image_path)
    image = image[50:128+50, 25:128+25, :]
    #image = image[25:25+178, 0:178, :]
    return transform(image, input_height, input_width, resize_height, resize_width, byte_image=byte_image)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
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
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, byte_image=True):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    if byte_image:
        return np.array(cropped_image).astype(np.uint8)
    else:
        return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.