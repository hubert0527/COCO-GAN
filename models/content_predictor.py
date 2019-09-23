import tensorflow as tf
from ops import batch_norm, lrelu, snlinear

from models.model_base import Model

_EPS = 1e-5

class ContentPredictorBuilder(Model):
    def __init__(self, config):
        self.config=config
        self.z_dim = config["model_params"]["z_dim"]
        self.aux_dim = config["model_params"]["aux_dim"]
        self.update_collection = "Q_update_collection"
            
    def forward(self, h, is_training):
        print(" [Build] Spatial Predictor ; is_training: {}".format(is_training))
        update_collection = self._get_update_collection(is_training)
        with tf.variable_scope("Q_content_prediction_head", reuse=tf.AUTO_REUSE):
            h = snlinear(h, self.aux_dim, 'fc1', update_collection=update_collection)
            h = batch_norm(name='bn1')(h, is_training=is_training)
            h = lrelu(h)
            h = snlinear(h, self.z_dim, 'fc2', update_collection=update_collection)
            return tf.nn.tanh(h)
