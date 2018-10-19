from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops

import gac
from gac.tf.hyperparameters import BATCH_SIZE, IMAGE_SIZE

def inputs(tfrecord_file, batch_size=None, num_epochs=1):
    """Generate batches of images for evaluation.

    Use this function as the inputs for evaluating a network.

    Note that some (minimal) image preprocessing occurs during evaluation
    including central cropping and resizing of the image to fit the network.

    Args:
      tfrecord_file: tfrecord file
      batch_size: integer, number of examples in batch
      num_preprocess_threads: integer, total number of preprocessing threads but
        None defaults to FLAGS.num_preprocess_threads.

    Returns:
      images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                         image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
    """
    if not batch_size:
        batch_size = BATCH_SIZE

    images, labels = batch_inputs(
        tfrecord_file, batch_size,
        num_epochs=num_epochs)

    return images, labels


def decode_png(image_buffer, channels=3, scope=None):
    """Decode a PNG/JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_png'):
        image = tf.image.decode_png(image_buffer, channels=channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def image_preprocessing(image_buffer, width, height):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image_buffer: PNG/JPEG encoded string Tensor
      width: Width of the image
      height: Height of the image

    Returns:
      3-D float Tensor containing an appropriately scaled image
    """
    image = decode_png(image_buffer)
    image = tf.image.resize_images(image, [height, width])

    return image


def audio_to_spectrogram(
        audio_buffer,
        width,
        height,
        channels=1,
        window_size=1024,
        stride=64,
        brightness=100.,
        scope=None):
    """Decode and build a spectrogram using a wav string tensor.

    Args:
      audio_buffer: Audio encoded string tensor.
      width: Spectrogram width.
      height: Spectrogram height.
      channels: Channels count.
      window_size: Size of the spectrogram window.
      stride: Size of the spectrogram stride.
      brightness: Brightness of the spectrogram.
      scope: Optional scope name.

    Returns:
      3-D float tensor spectrogram.
    """
    with tf.name_scope(values=[audio_buffer], name=scope,
                       default_name='audio_to_spectrogram'):
        waveform = audio_ops.decode_wav(
            audio_buffer, desired_channels=channels)
        spectrogram = audio_ops.audio_spectrogram(
            waveform.audio,
            window_size=window_size,
            stride=stride)
        brightness = tf.constant(brightness)
        mul = tf.multiply(spectrogram, brightness)
        min_const = tf.constant(255.)
        minimum = tf.minimum(mul, min_const)
        expand_dims = tf.expand_dims(minimum, -1)
        resize = tf.image.resize_bilinear(expand_dims, [width, height])
        squeeze = tf.squeeze(resize, 0)
        flip_left_right = tf.image.flip_left_right(squeeze)
        transposed = tf.image.transpose_image(flip_left_right)
        cast = tf.cast(transposed, tf.uint8)
        image = tf.image.encode_png(cast)

        return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a tf.Example

    The output of the build_tfrecords.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/class/label: 0
      image/class/text: b'machine'
      image/filename: b'f343sdaw.png'
      image/encoded: <PNG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a PNG file.
      label: Tensor tf.int32 containing the label.
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    features = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }

    features = tf.parse_single_example(example_serialized, features=features)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label, features['image/class/text']


def batch_inputs(tfrecords_file, batch_size=BATCH_SIZE, num_epochs=1):
    """Contruct batches of training or evaluation examples from a tfrecord file.

    Args:
      tfrecords_file: The tfrecords file to read
      batch_size: integer
      num_epochs: integer

    Returns:
      images: 4-D float Tensor of a batch of images
      labels: 1-D integer Tensor of [batch_size].

    Raises:
      ValueError: if tfrecords_file is not found
    """

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        with tf.name_scope('batch_processing'):
            if not os.path.exists(tfrecords_file):
                raise FileNotFoundError(
                    'Tfrecord file %s not found' %
                    tfrecords_file)

            # Create filename_queue
            filename_queue = tf.train.string_input_producer(
                [tfrecords_file], num_epochs=num_epochs)

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            image_buffer, label, _ = parse_example_proto(serialized_example)

            # Reshape images into these desired dimensions.
            height = IMAGE_SIZE
            width = IMAGE_SIZE

            image = image_preprocessing(image_buffer, width, height)

            # Creates batches by randomly shuffling tensors
            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=batch_size,
                                                    capacity=7000,
                                                    num_threads=1,
                                                    min_after_dequeue=batch_size * 3)

            gac.tf.utils.images_summary(images)

            return images, tf.reshape(labels, [batch_size])
