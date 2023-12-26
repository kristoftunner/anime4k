import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal

# Modified depth_to_space shuffle order for easier shader generation


class DepthToSpace2(tf.keras.layers.Layer):
    def __init__(self, input_depth, **kwargs):
        super(DepthToSpace2, self).__init__(**kwargs)
        self.input_depth = input_depth

    def build(self, input_shape):
        super(DepthToSpace2, self).build(input_shape)

    def call(self, x):
        x = tf.split(x, (self.input_depth // 4), axis=-1)
        return tf.concat([tf.nn.depth_to_space(xx, 2) for xx in x], axis=-1)

# SR model that doubles image size


def create_sr2_model(input_depth=3, highway_depth=4, block_depth=4, init='he_normal', init_last=RandomNormal(mean=0.0, stddev=0.001)):

    input_shape = [None, None, input_depth]
    input_lr = tf.keras.layers.Input(shape=input_shape)
    input_lr2 = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear')(input_lr)

    depth_list = []

    x = input_lr
    for i in range(block_depth):
        x = tf.keras.layers.Conv2D(
            highway_depth, (3, 3), padding='same', kernel_initializer=init)(x)
        x = tf.nn.crelu(x)
        depth_list.append(x)

    x = tf.keras.layers.Concatenate(axis=-1)(depth_list)
    x = tf.keras.layers.Conv2D(
        4*input_depth, (1, 1), padding='same', kernel_initializer=init_last)(x)
    x = DepthToSpace2(4*input_depth)(x)

    x = tf.keras.layers.Add()([x, input_lr2])

    model = tf.keras.models.Model(input_lr, x)

    return model
