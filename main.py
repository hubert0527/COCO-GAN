import yaml
import argparse

import tensorflow as tf

from models.generator import GeneratorBuilder
from models.discriminator import DiscriminatorBuilder
from models.spatial_prediction import SpatialPredictorBuilder
from models.content_predictor import ContentPredictorBuilder

from coord_handler import CoordHandler
from patch_handler import PatchHandler

from data_loader import DataLoader
from trainer import Trainer
from evaluator import Evaluator
from logger import Logger

from fid_utils import fid

def precompute_parameters(config):
    full_image_size = config["data_params"]["full_image_size"]
    micro_patch_size = config["data_params"]["micro_patch_size"]
    macro_patch_size = config["data_params"]["macro_patch_size"]

    # Let NxM micro matches to compose a macro patch,
    #    `ratio_macro_to_micro` is N or M
    ratio_macro_to_micro = [
        macro_patch_size[0] // micro_patch_size[0],
        macro_patch_size[1] // micro_patch_size[1],
    ]
    num_micro_compose_macro = ratio_macro_to_micro[0] * ratio_macro_to_micro[1]

    # Let NxM micro matches to compose a full image,
    #    `ratio_full_to_micro` is N or M
    ratio_full_to_micro = [
        full_image_size[0] // micro_patch_size[0],
        full_image_size[1] // micro_patch_size[1],
    ]
    num_micro_compose_full = ratio_full_to_micro[0] * ratio_full_to_micro[1]

    config["data_params"]["ratio_macro_to_micro"] = ratio_macro_to_micro
    config["data_params"]["ratio_full_to_micro"] = ratio_full_to_micro
    config["data_params"]["num_micro_compose_macro"] = num_micro_compose_macro
    config["data_params"]["num_micro_compose_full"] = num_micro_compose_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

        # Basic protect. Otherwise, I don't know what will happen. OuO
        micro_size = config["data_params"]['micro_patch_size']
        macro_size = config["data_params"]['macro_patch_size']
        full_size = config["data_params"]['full_image_size']
        assert macro_size[0] % micro_size[0] == 0
        assert macro_size[1] % micro_size[1] == 0
        assert full_size[0] % micro_size[0] == 0
        assert full_size[1] % micro_size[1] == 0

    # Pre-compute some frequently used parameters
    precompute_parameters(config)

    # Create model builders
    coord_handler = CoordHandler(config)
    patch_handler = PatchHandler(config)
    g_builder = GeneratorBuilder(config)
    d_builder = DiscriminatorBuilder(config)
    cp_builder = SpatialPredictorBuilder(config)
    zp_builder = ContentPredictorBuilder(config)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        # Build TF records
        real_images = DataLoader(config).build()

        # Create controllers
        trainer = Trainer(sess, config, real_images, 
                          g_builder, d_builder, cp_builder, zp_builder, 
                          coord_handler, patch_handler)
        evaluator = Evaluator(sess, config)
        logger = Logger(sess, config, patch_handler)

        # Build graphs
        print(" [Build] Constructing training graph...")
        trainer.build_graph()
        print(" [Build] Constructing evaluation graph...")
        evaluator.build_graph()
        print(" [Build] Constructing logging graph...")
        logger.build_graph(trainer)

        # Initialize all variables
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(coord=tf.train.Coordinator())

        # Load checkpoint
        global_step = logger.load_ckpt()

        # Start training
        trainer.train(logger, evaluator, global_step)
        





        



    
