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
from typing import Any

import keras
from keras import ops

import numpy as np
from numpy import typing as npt

import yaml

from src.data import DataGenerator, get_audio_files
from src.encoder_types import Config
from src.post_processing import calculate_similarity_matrix, get_song_encodings, print_matrix



LOAD_FROM_FILE: None | str = None

#
# Hyperparameters
#

EPOCHS = 20
BATCH_SIZE = 64
MARGIN = 1  # Margin for contrastive loss.
LATENT_SPACE = 20
TRAIN_TEST_SPLIT = 0.8
STEPS_PER_EPOCH = 3000
# if this is too hight, we get some sort of "rolling overfitting"
#   where the accuracy stays hight because the data is too samey for too long
# but if this is too long, it takes too long to load
SAMPLES_PER_SONG_LOAD = 1000

AUDIO_FOLDER = r'D:\song_encoder\audio'

CONFIG: Config = yaml.load(
    open('config.yaml'),
    yaml.Loader,
)

##### HERE'S THE OUTPUTS THAT MATTER TO YOU #####
# data input rate = (EXPECTED_BITRATE / SPECTROGRAM_FRAME_GROUPING) frames per second
#   i've tried to go for ~30 because it "feels right"
# print(EXPECTED_BITRATE / SPECTROGRAM_FRAME_GROUPING)

# image width in seconds = IMAGE_WIDTH * SPECTROGRAM_FRAME_GROUPING * SPECTROGRAM_NPERSEG / EXPECTED_BITRATE
#   this is ~12 seconds for no real reason
# print(IMAGE_WIDTH * SPECTROGRAM_FRAME_GROUPING * SPECTROGRAM_NPERSEG / EXPECTED_BITRATE)

# OTHER IMPORTANT VALUES
# MAX_EUCLIDEAN_DISTANCE = (4*LATENT_SPACE)**0.5



# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(x: keras.KerasTensor, y: keras.KerasTensor) -> Any:
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(sum_square)


"""
## Define the contrastive Loss
"""


def pre_loss(vects: tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]) -> Any:
    x, y, z = vects

    same_dist = euclidean_distance(x, y)
    diff_dist = ops.minimum(
        euclidean_distance(x, z),
        MARGIN,
    )
    output = ops.concatenate((same_dist, diff_dist), 1)
    return output



def build_embedded_network(model_name: str = 'embedding_network') -> keras.Model:
    embedding_network = keras.Sequential(
        [
            keras.layers.Input((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 1)),
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

            keras.layers.Conv2D(24, (5, 5), strides=(2,2), activation=keras.layers.LeakyReLU(0.1)),
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
        ],
        name=model_name,
    )

    return embedding_network


def build_siamese_model(embedding_network: keras.Model) -> keras.Model:
    input_1 = keras.layers.Input((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 1))
    input_2 = keras.layers.Input((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 1))
    input_3 = keras.layers.Input((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 1))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)
    tower_3 = embedding_network(input_3)

    output_layer = keras.layers.Lambda(pre_loss, output_shape=(1,))(
        [tower_1, tower_2, tower_3]
    )
    # normal_layer = keras.layers.BatchNormalization()(merge_layer)
    # output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2, input_3], outputs=output_layer)

    return siamese



print('preparing data')
input_filepaths = get_audio_files(AUDIO_FOLDER)

# from scipy.io import wavfile
# for f in list(input_filepaths):
#     if wavfile.read(f)[0] != CONFIG['EXPECTED_BITRATE']:
#         input_filepaths.remove(f)

random.shuffle(input_filepaths)

max_test = min(
    int(len(input_filepaths)*(1-TRAIN_TEST_SPLIT)),
    20,
)

test_filepaths = input_filepaths[:max_test]
train_filepaths = input_filepaths[max_test:]


train_data = DataGenerator(
    train_filepaths,
    CONFIG,
    batch_size=BATCH_SIZE,
    samples_per_song_load=SAMPLES_PER_SONG_LOAD,
    num_active_files=15,
)
test_data = DataGenerator(
    test_filepaths,
    CONFIG,
    batch_size=BATCH_SIZE,
    samples_per_song_load=SAMPLES_PER_SONG_LOAD,
    num_active_files=2,
)



embedded_model: keras.Model
if LOAD_FROM_FILE is None:
    embedding_network = build_embedded_network()
    siamese = build_siamese_model(embedding_network)

    siamese.compile(loss=keras.losses.MeanSquaredError(), optimizer="RMSprop", metrics=["accuracy"])
    siamese.summary()
    embedding_network.summary()


    print()
    print('training on', len(train_filepaths), 'songs')
    print('testing on', len(test_filepaths), 'songs')
    print()

    checkpoint = keras.callbacks.ModelCheckpoint('out/models/checkpoint_{epoch:02d}.keras', save_freq='epoch')
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

    embedded_layer = siamese.get_layer('embedding_network')
    embedded_model = keras.Model(inputs = embedded_layer.input, outputs = embedded_layer.output)

    embedded_model.save('out/models/out.keras')

else:
    embedded_model = keras.saving.load_model(LOAD_FROM_FILE)


all_filepaths = test_filepaths[:80] + train_filepaths[:20]

encodings = {}

for i, filename in enumerate(all_filepaths):
    print(f'{100*i/len(all_filepaths): >6.2f}%', end='\b'*7, flush=True)
    encodings[filename] = get_song_encodings(embedded_model, filename, CONFIG, 400)

a = calculate_similarity_matrix(
    encodings,
    test_filepaths[:20],
    samples_per_song=3000,
)
print()
print("test confusion")
print_matrix(a, 'avg_from_mean')
print()
print()

b = calculate_similarity_matrix(
    encodings,
    train_filepaths[:20],
    samples_per_song=3000,
)
print("train confusion")
print_matrix(b, 'avg_from_mean')

print()
print()

c = calculate_similarity_matrix(
    encodings,
    all_filepaths,
    samples_per_song=1500,
)
print("similarity matrix")
print_matrix(c, 'dist_mean')


