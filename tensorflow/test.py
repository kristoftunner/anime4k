import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import cv2
import time
import tf2onnx
import logging
from model import create_sr2_model


class DepthToSpace2(tf.keras.layers.Layer):
    def __init__(self, input_depth, **kwargs):
        super(DepthToSpace2, self).__init__(**kwargs)
        self.input_depth = input_depth

    def build(self, input_shape):
        super(DepthToSpace2, self).build(input_shape)

    def call(self, x):
        x = tf.split(x, (self.input_depth // 4), axis=-1)
        return tf.concat([tf.nn.depth_to_space(xx, 2) for xx in x], axis=-1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Load the model
    K.reset_uids()
    init_last = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

    model = create_sr2_model(input_depth=3, highway_depth=3, block_depth=2)
    model.summary()
    model.load_weights("tensorflow/anime4k-3-3-2.h5")

    # model.compile()
    image = cv2.imread("tensorflow/wallpaper1.jpg")
    image = cv2.resize(image, (1920, 1280))
    cv2.imwrite("tensorflow/wallpaper1_base.jpg", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input = np.expand_dims(image, axis=0)
    input = input / 255.0
    input = tf.convert_to_tensor(input, dtype=tf.float32)

    start_time = time.time()
    output = model.predict(input)
    end_time = time.time()
    logging.info(f"Execution time: {end_time - start_time}")
    output = (np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)[0]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tensorflow/output.jpg", output)

    # input_spec = (tf.TensorSpec(
    #    (1, 1280, 1920, 3), tf.float32, name="input"),)
    # tf2onnx.convert.from_keras(model, input_signature=input_spec,
    #                           output_path="anime4k-xs.onnx", opset=13)
