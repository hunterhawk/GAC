#!/usr/bin/env python
"""Export amd model given existing training checkpoints

The model is exported with proper signatures that can be loaded by standard
tensorflow_model_server.
"""

from __future__ import print_function

import os.path

from tensorflow.contrib.framework.python.ops import audio_ops
import tensorflow as tf

import amd
from amd.tf.preprocessing import audio_to_spectrogram, image_preprocessing
from amd.tf.hyperparameters import *

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/amd-train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/amd-export',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', IMAGE_SIZE,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS


def export():
    with tf.Graph().as_default():
        # run preprocessing on cpu
        with tf.device('/cpu:0'):
            # Input transformation.
            serialized_tf_example = tf.placeholder(
                tf.string, name='tf_example')
            feature_configs = {
                'audio/wav': tf.FixedLenFeature(
                    shape=[], dtype=tf.string),
            }
            tf_example = tf.parse_single_example(
                serialized_tf_example, feature_configs)
            encoded_audio = tf_example['audio/wav']

            spectrogram = audio_to_spectrogram(encoded_audio,
                                               IMAGE_SIZE, IMAGE_SIZE)
            image = image_preprocessing(spectrogram, IMAGE_SIZE, IMAGE_SIZE)

            # expand dims so we get a 3D tensor
            images = tf.expand_dims(image, 0)

        # Disable dropout during inference
        dropout = tf.constant(1.0)

        # Run inference
        logits = amd.tf.ops.inference(images, dropout)

        # Transform output to topK result.
        values, indices = tf.nn.top_k(logits, NUM_CLASSES)

        # cast indices to int64
        indices = tf.to_int64(indices)

        # Apply softmax
        normalized_values = tf.nn.softmax(values)

        # Create a constant string Tensor where the i'th element is
        # the human readable class description for the i'th index.
        class_descriptions = amd.samples.get_labels()
        class_tensor = tf.constant(class_descriptions)

        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            class_tensor)

        classes = table.lookup(indices)

        # Restore variables from training checkpoint.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            # get checkpoint state from checkpoint dir.
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint.
                saver.restore(sess, ckpt.model_checkpoint_path)

                # Assuming model_checkpoint_path looks something like:
                #   /tmp/amd-train/model.ckpt-230,
                # extract global_step 230 from it.
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Export inference model.
            output_path = os.path.join(
                tf.compat.as_bytes(FLAGS.output_dir),
                tf.compat.as_bytes('amd'),
                tf.compat.as_bytes(str(FLAGS.model_version)))
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            # Build the signature_def_map.
            classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
                serialized_tf_example)
            classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                classes)
            scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                normalized_values)
            labels_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                indices
            )

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                            classify_inputs_tensor_info
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classes_output_tensor_info,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                            scores_output_tensor_info
                    },
                    method_name=tf.saved_model.signature_constants.
                    CLASSIFY_METHOD_NAME))

            predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
                encoded_audio)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'images': predict_inputs_tensor_info},
                    outputs={
                        'classes': classes_output_tensor_info,
                        'labels': labels_output_tensor_info,
                        'scores': scores_output_tensor_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(
                tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            print('Successfully exported model to %s' % output_path)


def main(unused_argv=None):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    export()


if __name__ == '__main__':
    tf.app.run()
