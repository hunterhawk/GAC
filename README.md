# GAC

This repository contains the GAC(General Audio Classification) project. My goal is to develop an e2e arquitecture for a general audio classifier that is both powerful and scalable.

## Installation

First of all you need to install the [Bazel](https://docs.bazel.build/versions/master/install.html) build system and have a working python 2/3 interpreter.

Clone the repository:

```bash
git clone repo_here
```

Let's install the python deps. We are going to use `pipenv` to create a fresh environment:

```bash
pip install pipenv
pipenv sync
```

If you have a NVIDIA GPU available, install CUDA and CuDNN to take advantage of the GPU during the training process. Download CUDA 9.0 along with CuDNN 7.3 (compiled for CUDA 9).

> If you wish to train using a CPU, remove tensorflow-gpu and install tensorflow: `pip uninstall tensorflow-gpu && pip install tensorflow`. Keep in mind that the training process will take much longer (6 to 10 times longer depending on your CPU).

## Downloading the dataset

I choose the [ESC-50] dataset to train and validate the model. Download the dataset .zip (600mb) and extract it into the `ESC-50` directory.

## Building and Running

You can build the targets using [Bazel](http://bazel.build)

```bash
bazel build //scripts/... # build all targets in scripts folder
bazel build //gac:lib # build the gac library
bazel build //scripts:gac_train # build the train script
bazel build //scripts:gac_eval # build the eval script
bazel build //scripts:gac_build_tfrecords # build the tfrecords script
bazel build //scripts:gac_saved_model # build the tensorflow serving exporter script
```

Run the generated targets:

```bash
# build tfrecords containing train and test set
./bazel-bin/scripts/gac_build_tfrecords

# train on train set
./bazel-bin/scripts/gac_train

# evaluate on test set
./bazel-bin/scripts/gac_eval

# export the model
./bazel-bin/scripts/gac_saved_model
```
