import tensorflow as tf
import numpy as np
import os
import time


def reuse(scope):
    """
    A decorator for transparent reuse of tensorflow
    `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_ in a
    function. The decorated function will automatically create variables the
    first time they are called and reuse them thereafter.
    .. note::
        This decorator is internally implemented by tensorflow's
        :func:`make_template` function. See `its doc
        <https://www.tensorflow.org/api_docs/python/tf/make_template>_`
        for requirements on the target function.
    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    """
    return lambda f: tf.make_template(scope, f)


class Batches:
    def __init__(self, X, batch_size):
        self.X     = X
        self.start = 0
        self.batch_size = batch_size

    def _call(self):
        ret = self.X[self.start:self.start+self.batch_size]
        self.start = min(self.start+self.batch_size, self.X.shape[0])
        if self.start == self.X.shape[0]:
            self.start = 0
        return ret
