import os
import tensorflow as tf

from utils import aug_cylindrical_data_tensor

class DataLoader():
    def __init__(self, config):
        self.config = config
        self.batch_size = config["train_params"]["batch_size"]
        self.input_size = config["data_params"]["full_image_size"]
        self.c_dim = self.config["data_params"]["c_dim"]
        self.dataset = config["data_params"]["dataset"].lower()
        self.coordinate_system = config["data_params"]["coordinate_system"]

        assert self.dataset in {
            "celeba",
            "celeba-syn-inward",
            "celeba-syn-outward",
            "celeba-hq",
            "lsun",
            "mp3d",
        }, "Specified dataset `{}` is not supported!".format(self.dataset)

    def _read_and_decode(self, tfrecords_path):
        filename_queue = tf.train.string_input_producer([tfrecords_path])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                            features={
                                                'img' : tf.FixedLenFeature([], tf.string),
                                            })
        img = tf.decode_raw(features['img'], tf.uint8)
        img = tf.reshape(img, [self.input_size[0], self.input_size[1], self.c_dim])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        return img

    def build(self):
        tfrecords_path = "./tfrecords/{}_{}x{}.tfrecords".format(self.dataset, self.input_size[0], self.input_size[1])
        if not os.path.exists(tfrecords_path):
            raise ValueError("Please generate TFRecords first, and place it at: \n\t{}".format(tfrecords_path))
        image_t = self._read_and_decode(tfrecords_path)
        batch_images_t = tf.train.shuffle_batch([image_t], batch_size=self.batch_size,
                                        capacity=self.batch_size*8, num_threads=min(self.batch_size//4,16), 
                                        min_after_dequeue=self.batch_size*2)
        if self.coordinate_system=="cylindrical":
            print(" [*] Data with cylindrical coordinate auto applies z-axis-rotation augmentation.")
            batch_images_t  = aug_cylindrical_data_tensor(batch_images_t)
        return batch_images_t
