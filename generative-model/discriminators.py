import tensorflow as tf
from tensorflow.contrib import layers
from utils import reuse

def get_discriminator(dataset, arch, n_x, n_xl, n_channels, n_z, ndf, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None,
                         'decay': 0.9}

    if dataset == 'mnist':
        discriminator = None
    else:
        @reuse('discriminator')
        def discriminator(x):
            print(x.get_shape())
            h = layers.conv2d(x, ndf*2, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, ndf*4, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.conv2d(h, ndf*8, 5, stride=2,
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.flatten(h)
            h = layers.fully_connected(h, n_z, activation_fn=None)
            return h

    return discriminator
