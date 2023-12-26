#!/home/kristoft/miniconda3/bin/python
from model import *
import tensorflow as tf
from PIL import Image
import os
import logging
from matplotlib import pyplot as plt
from utils import degrade_ring, degrade_blur_gaussian, degrade_noise_gaussian, degrade_rgb_to_yuv, degrade_yuv_to_rgb, show_images
import numpy as np


def augment_images(img):
    img = img / 255

    img = tf.image.random_hue(img, 0.5)
    img = tf.image.random_contrast(img, 0.5, 2.0)
    img = tf.clip_by_value(img, 0, 1)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.rot90(
        img, k=tf.experimental.numpy.random.randint(4, dtype=tf.int32))

    if tf.random.uniform(shape=()) < 0.1:
        img = degrade_blur_gaussian(img, 1.0, shape=(5, 5))

    lr, hr = img, img

    if tf.random.uniform(shape=()) < 0.1:
        random_sigma = tf.random.uniform(shape=(), minval=2.0, maxval=5.0)
        lr = degrade_ring(lr, random_sigma, shape=(5, 5))

    if tf.random.uniform(shape=()) < 0.1:
        random_sigma = tf.random.uniform(shape=(), minval=0.1, maxval=0.5)
        lr = degrade_blur_gaussian(lr, random_sigma, shape=(3, 3))

    hr_shape = tf.shape(hr)
    if tf.random.uniform(shape=()) < 0.5:
        lr = tf.image.resize(
            lr, [hr_shape[-3]//2, hr_shape[-2]//2], method="area")
    else:
        lr = tf.image.resize(
            lr, [hr_shape[-3]//2, hr_shape[-2]//2], method="bicubic")

    if tf.random.uniform(shape=()) < 0.8:
        lr = degrade_rgb_to_yuv(lr, jpeg_factor=tf.experimental.numpy.random.randint(
            70, 90, dtype=tf.int32), chroma_subsampling=True, chroma_method="area")
        lr = degrade_yuv_to_rgb(lr, chroma_method="bicubic")
        # Process hr alongside with lr to prevent mean shift from jpeg and conversion errors
        hr = degrade_rgb_to_yuv(hr, jpeg_factor=95, chroma_subsampling=False)
        hr = degrade_yuv_to_rgb(hr)

    return lr, hr


def augment_images_valid(img):
    img = img / 255

    lr, hr = img, img

    hr_shape = tf.shape(hr)
    lr = tf.image.resize(lr, [hr_shape[-3]//2, hr_shape[-2]//2], method="area")

    return lr, hr

# YUV loss to weigh in favour of luminance (2 to 1), as humans are less sensitive to chroma degradation


def load_dataset(dir) -> np.ndarray:
    images = []
    for file in os.listdir(dir):
        image = Image.open(os.path.join(dir, file))
        image_array = np.array(image)
        images.append(image_array)

    images_np = np.array(images)
    return images_np


def concat_images(images, concat_max=5) -> np.ndarray:
    concatenated_image = images[0]
    for i in range(1, min(concat_max, images.shape[0])):
        concatenated_image = np.concatenate(
            (concatenated_image, images[i]), axis=1)
    return concatenated_image


def show_sample(sample):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(concat_images(sample[0], 5))
    plt.subplot(2, 1, 2)
    plt.imshow(concat_images(sample[1], 5))
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("TensorFlow version: %s", tf.__version__)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Detected GPUs:", len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # create model
    model = create_sr2_model(input_depth=3, highway_depth=1, block_depth=1)
    model.summary()

    dataset_signature = tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8)

    synla_4096_dataset = load_dataset(
        "/home/kristoft/dev/repos/datasets/Dataset_4096")
    images_synla_4096 = tf.data.Dataset.from_generator(
        lambda: synla_4096_dataset, output_signature=(dataset_signature))
    logging.info(f"Loaded {len(synla_4096_dataset)} images")

    synla_1024_dataset = load_dataset(
        "/home/kristoft/dev/repos/datasets/Dataset_1024")
    images_synla_1024 = tf.data.Dataset.from_generator(
        lambda: synla_1024_dataset, output_signature=(dataset_signature))
    logging.info(f"Loaded {len(synla_1024_dataset)} images")

    batch_size = 32
    dataset_train = images_synla_4096
    dataset_train = dataset_train.map(
        augment_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

    dataset_valid = images_synla_1024
    dataset_valid = dataset_valid.map(
        augment_images_valid, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.batch(batch_size)
    dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)

    training_sample = next(iter(dataset_train))
    show_sample(training_sample)
    MSE_Loss = tf.keras.losses.MeanSquaredError()

    def YUV_Error(y_true, y_pred):
        true_yuv = tf.image.rgb_to_yuv(y_true)
        pred_yuv = tf.image.rgb_to_yuv(y_pred)

        true_y, true_u, true_v = tf.split(true_yuv, 3, axis=-1)
        pred_y, pred_u, pred_v = tf.split(pred_yuv, 3, axis=-1)

        y_err = MSE_Loss(true_y, pred_y) * 0.5
        u_err = MSE_Loss(true_u, pred_u) * 0.25
        v_err = MSE_Loss(true_v, pred_v) * 0.25

        return (y_err + u_err + v_err)

    # Super-convergence with clipping followed by fine tuning with Adam allows somewhat fair convergence within a few minutes
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=10000.0,
                  clipvalue=0.00000001, momentum=0.9, decay=0.0, nesterov=True), loss=YUV_Error)
    model.fit(dataset_train.repeat(), epochs=1,
              steps_per_epoch=4096, validation_data=dataset_valid)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss=YUV_Error)
    model.fit(dataset_train.repeat(), epochs=1,
              steps_per_epoch=4096*2, validation_data=dataset_valid)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), loss=YUV_Error)
    model.fit(dataset_train.repeat(), epochs=1,
              steps_per_epoch=4096*2, validation_data=dataset_valid)

    # Show results
    validation_sample = next(iter(dataset_valid))
    show_sample(validation_sample)

    pred_pred = model.predict(validation_sample[0])
    resized_valid_sample = tf.image.resize(
        validation_sample[0], [validation_sample[1].shape[1], validation_sample[1].shape[2]], method="bilinear")

    model.save_weights("anime4k-1-1.h5")
