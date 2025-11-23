"""
Title: Image similarity estimation using a Siamese Network with a contrastive loss
Author: Mehdi
Date created: 2021/05/06
Last modified: 2022/09/10
Description: Similarity learning using a siamese network trained with a contrastive loss.
Accelerator: GPU
"""

"""
## Introduction

[Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)
are neural networks which share weights between two or more sister networks,
each producing embedding vectors of its respective inputs.

In supervised similarity learning, the networks are then trained to maximize the
contrast (distance) between embeddings of inputs of different classes, while minimizing the distance between
embeddings of similar classes, resulting in embedding spaces that reflect
the class segmentation of the training inputs.
"""

"""
## Setup
"""

print('importing')
import random
from typing import Callable, Any, Optional

import keras
from keras import ops
from keras import layers

import numpy as np
from numpy import typing as npt

from src.data import DataGenerator, get_audio_files
from src.post_processing import i_am_confusion, print_matrix


from src.encoder_types import Config_Type



LOAD_FROM_FILE: None | str = None

#
# Hyperparameters
#

EPOCHS = 1
BATCH_SIZE = 64
MARGIN = 1  # Margin for contrastive loss.
LATENT_SPACE = 20
TRAIN_TEST_SPLIT = 0.8
STEPS_PER_EPOCH = 3000
# if this is too hight, we get some sort of "rolling overfitting"
#   where the accuracy stays hight because the data is too samey for too long
# but if this is too long, it takes too long to load
SAMPLES_PER_SONG_LOAD = 1000

AUDIO_FOLDER = 'data/audio'

# don't process any file that uses a different bitrate
EXPECTED_BITRATE = 44100
# the raw window to calculate the spectrogram over (as a section of the bitrate)
#   changing this will also change the number of frequency buckets returned from scipy
SPECTROGRAM_NPERSEG = 128
# number of raw spectrogram frames to average over
#   nperseg will change the
SPECTROGRAM_FRAME_GROUPING = 12  # ~1500 from bitrate
# how many frames to input to the learning model
IMAGE_WIDTH = 360  # ~13 seconds

# i'm so sorry
#   you gotta find this by trial and error
# IMAGE_HEIGHT = load_data('file.wav', SPECTROGRAM_NPERSEG, SPECTROGRAM_FRAME_GROUPING).shape[0]
IMAGE_HEIGHT = 65

IMAGE_SIZE = IMAGE_HEIGHT, IMAGE_WIDTH

##### HERE'S THE OUTPUTS THAT MATTER TO YOU #####
# data input rate = (EXPECTED_BITRATE / SPECTROGRAM_FRAME_GROUPING) frames per second
#   i've tried to go for ~30 because it "feels right"
# print(EXPECTED_BITRATE / SPECTROGRAM_FRAME_GROUPING)

# image width in seconds = IMAGE_WIDTH * SPECTROGRAM_FRAME_GROUPING * SPECTROGRAM_NPERSEG / EXPECTED_BITRATE
#   this is ~12 seconds for no real reason
# print(IMAGE_WIDTH * SPECTROGRAM_FRAME_GROUPING * SPECTROGRAM_NPERSEG / EXPECTED_BITRATE)

# OTHER IMPORTANT VALUES
# MAX_EUCLIDEAN_DISTANCE = (4*LATENT_SPACE)**0.5

config: Config_Type = {
    'EPOCHS':EPOCHS,
    'BATCH_SIZE':BATCH_SIZE,
    'MARGIN':MARGIN,
    'LATENT_SPACE':LATENT_SPACE,
    'TRAIN_TEST_SPLIT':TRAIN_TEST_SPLIT,
    'STEPS_PER_EPOCH':STEPS_PER_EPOCH,
    'SAMPLES_PER_SONG_LOAD':SAMPLES_PER_SONG_LOAD,
    'AUDIO_FOLDER':AUDIO_FOLDER,
    'EXPECTED_BITRATE':EXPECTED_BITRATE,
    'SPECTROGRAM_NPERSEG':SPECTROGRAM_NPERSEG,
    'SPECTROGRAM_FRAME_GROUPING':SPECTROGRAM_FRAME_GROUPING,
    'IMAGE_WIDTH':IMAGE_WIDTH,
    'IMAGE_HEIGHT':IMAGE_HEIGHT,
    'IMAGE_SIZE':IMAGE_SIZE,
}

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects: tuple[keras.KerasTensor, keras.KerasTensor]) -> Any:
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


"""
## Define the contrastive Loss
"""


def loss(margin: float = MARGIN) -> Callable[..., Any]:
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(
            y_true: npt.NDArray[np.float32],
            y_pred: npt.NDArray[np.float32]
            ) -> Any:
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss



print('preparing data')
input_filepaths = get_audio_files(AUDIO_FOLDER)
# from scipy.io import wavfile
# illegal_files = [
#     f
#     for f in input_filepaths
#     if wavfile.read(f)[0] != EXPECTED_BITRATE
# ]
random.shuffle(input_filepaths)
# input_filepaths = input_filepaths[:10]
max_test = min(
    int(len(input_filepaths)*(1-TRAIN_TEST_SPLIT)),
    20
)
test_filepaths = input_filepaths[:max_test]
train_filepaths = [fpath for fpath in input_filepaths if fpath not in test_filepaths]

train_data = DataGenerator(
    train_filepaths,
    IMAGE_WIDTH,
    batch_size=BATCH_SIZE,
    dim=IMAGE_SIZE,
    spectrogram_nperseg=SPECTROGRAM_NPERSEG,
    spectrogram_frame_grouping=SPECTROGRAM_FRAME_GROUPING,
    samples_per_song_load=SAMPLES_PER_SONG_LOAD,
    num_active_files=15,
)
test_data = DataGenerator(
    test_filepaths,
    IMAGE_WIDTH,
    batch_size=BATCH_SIZE,
    dim=IMAGE_SIZE,
    spectrogram_nperseg=SPECTROGRAM_NPERSEG,
    spectrogram_frame_grouping=SPECTROGRAM_FRAME_GROUPING,
    samples_per_song_load=SAMPLES_PER_SONG_LOAD,
    num_active_files=2,
)


def build_embedded_network(model_name: str = 'embedding_network') -> keras.Model:
    embedding_network = keras.Sequential([
        keras.layers.Input((*IMAGE_SIZE, 1)),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(4, (5, 5), activation=keras.layers.LeakyReLU(0.1)),
        keras.layers.MaxPooling2D(pool_size=(2, 3)),
        keras.layers.Dropout(0.1),

        keras.layers.Conv2D(8, (5, 5), activation=keras.layers.LeakyReLU(0.1)),
        keras.layers.MaxPooling2D(pool_size=(1, 2)),
        keras.layers.Dropout(0.1),

        keras.layers.Conv2D(16, (5, 7), activation=keras.layers.LeakyReLU(0.1)),
        keras.layers.MaxPooling2D(pool_size=(1, 2)),
        keras.layers.Dropout(0.1),

        keras.layers.Conv2D(24, (5, 5), activation=keras.layers.LeakyReLU(0.1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(250, activation=keras.layers.LeakyReLU(0.1)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(200, activation=keras.layers.LeakyReLU(0.1)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(100, activation="tanh"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(LATENT_SPACE, activation="tanh"),
    ])

    return embedding_network


def build_siamese_model(embedding_network: keras.Model) -> keras.Model:
    input_1 = keras.layers.Input((*IMAGE_SIZE, 1))
    input_2 = keras.layers.Input((*IMAGE_SIZE, 1))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
        [tower_1, tower_2]
    )
    normal_layer = keras.layers.BatchNormalization()(merge_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    return siamese


embedding_network = build_embedded_network()
siamese = build_siamese_model(embedding_network)

siamese.compile(loss=keras.losses.MeanSquaredError(), optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()
embedding_network.summary()


"""
## Train the model
"""

print()
print('training on', len(train_filepaths), 'songs')
print('testing on', len(test_filepaths), 'songs')
print()

checkpoint = keras.callbacks.ModelCheckpoint('checkpoint_{epoch:02d}.keras', save_freq='epoch')
history = siamese.fit(
    x=train_data,
    validation_data=test_data,
    batch_size=BATCH_SIZE,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=300,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=[checkpoint],
)

siamese.save('out.keras')


a = i_am_confusion(
    siamese,
    test_filepaths[:20],
    config,
    batch_size=400,
    samples_per_song=3000,
)
print()
print("test confusion")
print_matrix(a[0])
print()
print()
print("test confidence")
print_matrix(a[1])
print()
print()
b = i_am_confusion(
    siamese,
    train_filepaths[:20],
    config,
    batch_size=400,
    samples_per_song=3000,
)
print("train confusion")
print_matrix(b[0])
print()
print()
print("train confidence")
print_matrix(b[1])
