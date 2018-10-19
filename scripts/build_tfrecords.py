#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf

from gac.tf.tfrecords import bytes_feature, int64_feature
from gac.esc50 import get_integer_label, read_files, get_human_label
from gac.tf.preprocessing import audio_to_spectrogram
from gac.tf.hyperparameters import *

tf.app.flags.DEFINE_string('output_dir', 'tfrecords/',
                           'Output data directory.')
tf.app.flags.DEFINE_float('train_size', 0.8,
                          'Percentage of train size.')

FLAGS = tf.app.flags.FLAGS


def _convert_to_example(filename, image_buffer, label, human):
    """Build an Example proto for an example.

    Args:
      filename: string, path to a wav file, e.g., '/path/to/example.wav'
      image_buffer: string, PNG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      human: string, human-readable label, e.g., 'class1, class2'
    Returns:
      Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': int64_feature(label),
        'image/class/text': bytes_feature(human),
        'image/filename': bytes_feature(os.path.basename(filename)),
        'image/encoded': bytes_feature(image_buffer)}))
    return example


class WavToSpectrogram(object):
    """Helper class that provides TensorFlow wav coding utilities."""

    def __init__(self):
        # Create a single Session to run all wav coding calls.
        self._sess = tf.Session()
        self._wav_data = tf.placeholder(dtype=tf.string)
        self._spectrogram = audio_to_spectrogram(
            self._wav_data, IMAGE_SIZE, IMAGE_SIZE)

    def generate_spectrogram(self, wav_data):
        image = self._sess.run(self._spectrogram,
                               feed_dict={self._wav_data: wav_data})
        return image


def _process_wav(filename, wav_to_spectrogram):
    """Process a single wav file.

    Args:
      filename: string, path to file e.g., '/path/to/example.wav'.
    Returns:
      An audio spectrogram encoded as a png image.
    """
    wav_data = tf.gfile.FastGFile(filename, 'rb').read()

    image = wav_to_spectrogram.generate_spectrogram(wav_data)

    return image


def _process_wav_files(filenames, labels, humans, type='train'):
    """Process and save list of wavs as TFRecord of Example protos.

    Args:
      filenames: list of strings; each string is a path to a wav file
      labels: list of integer; each integer identifies the ground truth
      humans: list of strings; each string is a human-readable label
    """
    assert len(filenames) == len(labels)
    assert len(filenames) == len(humans)
    assert type in ['train', 'test']

    wav_to_spectrogram = WavToSpectrogram()

    output_file = os.path.join(FLAGS.output_dir, type + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_file)
    for i, f in enumerate(filenames):
        label = labels[i]
        human = humans[i]

        image_buffer = _process_wav(f, wav_to_spectrogram)

        example = _convert_to_example(f, image_buffer, label,
                                      human)
        writer.write(example.SerializeToString())

        print('%d of %d - %s' % (i + 1, len(filenames), f))

    writer.close()
    print('Finished writing tfrecords to %s' % output_file)


def main(_):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # get the sample files
    filenames = read_files()
    labels = [get_integer_label(f) for f in filenames]
    humans = [get_human_label(f) for f in filenames]

    # split train_test
    train_samples = int(FLAGS.train_size * len(filenames))
    test_samples = int((1.0 - FLAGS.train_size) * len(filenames))

    train_filenames = filenames[0:train_samples]
    train_labels = labels[0:train_samples]
    train_humans = humans[0:train_samples]

    test_filenames = filenames[-test_samples:]
    test_labels = labels[-test_samples:]
    test_humans = humans[-test_samples:]

    print('Total samples: %d' % len(filenames))
    print('Train samples: %d' % train_samples)
    print('Test samples: %d' % test_samples)

    _process_wav_files(train_filenames, train_labels,
                       train_humans, 'train')
    _process_wav_files(test_filenames, test_labels,
                       test_humans, 'test')

    print('Results saved to %s' % FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
