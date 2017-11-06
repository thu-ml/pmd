import tensorflow as tf
from tensorflow.contrib import layers
from utils import reuse


def LeakyReLU(x):
    return tf.maximum(x, 0.2*x)


def get_discriminator(dataset, arch, n_x, n_xl, n_channels, n_z, ndf, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None,
                         'decay': 0.9}

    if dataset == 'mnist':
        @reuse('discriminator')
        def discriminator(x):
            h = layers.conv2d(x, ndf*2, 5, stride=2, activation_fn=LeakyReLU,
                                        biases_initializer=None)
            h = layers.conv2d(h, ndf*4, 5, stride=2, activation_fn=LeakyReLU,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, ndf*8, 5, padding='VALID', activation_fn=LeakyReLU,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, n_z, 3, padding='VALID', activation_fn=None,
                    biases_initializer=None)
            h = layers.flatten(h)
            return h
    else:
        @reuse('discriminator')
        def discriminator(x):
            h = layers.conv2d(x, ndf*2, 4, stride=2, activation_fn=LeakyReLU, biases_initializer=None)
            h = layers.conv2d(h, ndf*4, 4, stride=2, activation_fn=LeakyReLU,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, ndf*8, 4, stride=2, activation_fn=LeakyReLU,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, n_z, 4, padding='valid', activation_fn=None, biases_initializer=None)
            h = layers.flatten(h)
            return h

    return discriminator
