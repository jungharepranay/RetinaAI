"""
dataset_builder.py
-------------------
Build performant ``tf.data.Dataset`` pipelines with augmentation
for training and validation.
"""

import tensorflow as tf
from src.preprocessing import tf_decode_and_resize, IMG_SIZE

# ----- Augmentation layer (applied only during training) ----- #
_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")


def _load_sample(image_path, label):
    """Map function: decode image and pair with label."""
    img = tf_decode_and_resize(image_path, IMG_SIZE)
    return img, label


def _augment(image, label):
    """Apply augmentation (training only)."""
    image = _augmentation(image, training=True)
    return image, label


def build_dataset(image_paths, labels, batch_size=32,
                  shuffle=True, augment=True, buffer_size=1000):
    """
    Build a ``tf.data.Dataset`` from arrays of paths and labels.

    Parameters
    ----------
    image_paths : array-like of str
    labels : array-like of shape (N, 8)
    batch_size : int
    shuffle : bool
    augment : bool
    buffer_size : int

    Returns
    -------
    tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
