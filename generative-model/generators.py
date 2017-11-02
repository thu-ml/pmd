import tensorflow as tf
from tensorflow.contrib import layers
from utils import reuse

def get_generator(dataset, arch, n_x, n_xl, n_channels, n_z, ngf, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None,
                         'decay': 0.9}
    if arch == 'fc':
        @reuse('transformation')
        def generator(z_ph):
            h = layers.fully_connected(z_ph, 500, 
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.fully_connected(h, 500, 
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            x = layers.fully_connected(h, n_x, activation_fn=tf.nn.sigmoid)
            return tf.reshape(x, [-1, n_xl, n_xl, n_channels])
    elif arch == 'conv' or arch == 'adv':
        if dataset=='mnist':
            @reuse('transformation')
            def generator(z_ph):
                h = tf.reshape(z_ph, [-1, 1, 1, n_z])
                h = layers.conv2d_transpose(h, ngf*4, 3, padding='VALID',
                        normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
                h = layers.conv2d_transpose(h, ngf*2, 5, padding='VALID',
                        normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
                h = layers.conv2d_transpose(h, ngf, 5, stride=2, 
                        normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
                x = layers.conv2d_transpose(h, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
                return x 
        else:
            @reuse('transformation')
            def generator(z_ph):
                h = layers.fully_connected(z_ph, num_outputs=ngf*8*4*4,
                                           normalizer_fn=layers.batch_norm,
                                           normalizer_params=normalizer_params)
                h = tf.reshape(h, [-1, 4, 4, ngf*8])
                h = layers.conv2d_transpose(h, ngf*4, 5, stride=2,
                        normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
                h = layers.conv2d_transpose(h, ngf*2, 5, stride=2,
                        normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
                x = layers.conv2d_transpose(h, n_channels, 5, stride=2, activation_fn=tf.nn.sigmoid)
                return x
    else:
        @reuse('transformation')
        def generator(z_ph):
            h = layers.fully_connected(z_ph, 500, 
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            h = layers.fully_connected(h, 500, 
                    normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params)
            x = layers.fully_connected(h, n_x)  # TODO sigmoid?
            return x

    return generator
