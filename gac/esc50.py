import glob
import os
import random
import numpy as np
import pandas as pd

SPECTROGRAM_SIZE = (128, 128)
SPECTROGRAM_SIZE_WITH_CHANNELS = SPECTROGRAM_SIZE + (1,)

def read_csv_dataset():
    """Read ESC50 csv dataset .
    Returns:
        A pandas Dataframe
    """
    return pd.read_csv('ESC-50/meta/esc50.csv')

def get_labels():
    """Get the ESC50 labels.
    Returns:
        A list of string labels where the i'th element is 
        the human readable class description for the i'th index.
    """
    df = read_csv_dataset()
    return df.sort_values('target').category.unique()


def read_files(shuffle=True):
    """Read ESC-50 files.
    Args:
        shuffle: bool.
    Returns:
        An array containing the filepaths.
    """
    files = glob.glob('ESC-50/audio/*.wav')
    if shuffle:
        random.shuffle(files)
    return files


def get_human_label(file):
    """Get the human label for a file.
    Args:
        file: File path.
    Returns:
        String label.
    """
    _, filename = os.path.split(file)
    df = read_csv_dataset()
    return df.loc[df['filename'] == filename].iloc[0].category


def get_label(file):
    """Get the one hot encode label for this file.
    Args:
        file: File path.
    Returns:
        One Hot encode vector of labels like [0, 1, 0].
    """
    classes = get_labels()
    df = read_csv_dataset()
    index = get_integer_label(file)
    return np.eye(len(classes))[index]

def get_integer_label(file):
    """Get the integer label for a file.
    Args:
        file: File path.
    Returns:
        Integer label.
    """
    _, filename = os.path.split(file)
    df = read_csv_dataset()
    return df.loc[df['filename'] == filename].iloc[0].target
