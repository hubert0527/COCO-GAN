import tensorflow as tf
from ops import NO_OPS

class Model():
    
    def _get_update_collection(self, is_training):
        if is_training:
            return self.update_collection
        else:
            return NO_OPS

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)