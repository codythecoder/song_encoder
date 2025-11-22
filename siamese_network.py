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
import os
import random
from statistics import mean, stdev
import time
from typing import Callable, Optional, Any, TypeAlias

import keras
from keras import ops
from keras import layers

import numpy as np
from numpy import typing as npt

import matplotlib.pyplot as plt

from spectrogrammify import spectrogrammify

from collections import defaultdict

Image_Type: TypeAlias = npt.NDArray[np.float32]
Raw_Data_Point_Type: TypeAlias = tuple[tuple[Image_Type, Image_Type], int]


#
# Hyperparameters
#

EPOCHS = 1
BATCH_SIZE = 64
MARGIN = 1  # Margin for contrastive loss.
LATENT_SPACE = 20
TRAIN_TEST_SPLIT = 0.8
STEPS_PER_EPOCH = 5000

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


@keras.saving.register_keras_serializable()
class PatchScanner(layers.Layer):
    def __init__(
            self,
            patch_width: int,
            image_shape: tuple[int | None, ...],
            output_size: int | None,
            strides: Optional[int | tuple[int, int]] = None,
            layers: int = 1,
            **kwargs: dict[Any, Any]) -> None:
        super().__init__(**kwargs)
        sample_image_size: tuple[int | None, int | None, int | None] = (
            image_shape[1],
            image_shape[2],
            image_shape[3] if len(image_shape) >= 4 else 1
        )
        self.patch_size = (sample_image_size[0], patch_width)
        self.strides = strides
        self.layers = layers
        self.output_height: int | None = output_size or sample_image_size[0]


        input = keras.layers.Input((*self.patch_size, sample_image_size[2]))
        dense = keras.layers.Flatten()(input)
        dense = keras.layers.Dense(self.output_height*self.layers, activation=keras.layers.LeakyReLU(0.1))(dense)
        self.embedding_network = keras.Model(input, dense)

    def call(self, x: keras.KerasTensor) -> Any:
        # create patch virtulisation
        patches = keras.ops.image.extract_patches(x, self.patch_size, strides=self.strides)
        batch_size = keras.ops.shape(patches)[0]
        num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]
        num_dims = x.shape[-1]
        patches = keras.ops.reshape(patches, (batch_size, num_patches, *self.patch_size, num_dims))
        patches = keras.ops.transpose(patches, axes=(0, 2, 3, 4, 1))

        # fully connected layer factory
        y = keras.ops.unstack(
            patches, num=None, axis=-1
        )

        outputs = [
            layers.Reshape((self.output_height, 1, self.layers))(self.embedding_network(layer))
            for layer in y
        ]

        output = layers.Concatenate(2)(outputs)

        return output


def get_audio_files(audio_folder: str = AUDIO_FOLDER) -> list[str]:
    return [
        os.path.join(p, fname)
        for p, d, f in os.walk(audio_folder)
        for fname in f
        if fname.endswith('.wav')
    ]


def load_data(
        filepath: str,
        nperseg: int = 256,
        seg_length: int = 1,
        ) -> npt.NDArray[np.float64]:

    bitrate, _, _, spectrogram = spectrogrammify(filepath, nperseg=nperseg)
    if bitrate != EXPECTED_BITRATE:
        print(bitrate)
        print(filepath)
        assert bitrate == EXPECTED_BITRATE

    spectrogram = np.array([
        spectrogram[:,j*seg_length:j*seg_length+seg_length].max(axis=1)
        for j in range(len(spectrogram[0])//seg_length)
    ], dtype=np.float64).transpose((1, 0))

    spectrogram /= spectrogram.max()

    return spectrogram


class DataGenerator(keras.utils.PyDataset):
    def __init__(
            self,
            filenames: list[str],
            data_width: int,
            batch_size: int = BATCH_SIZE,
            dim: tuple[int, ...] = IMAGE_SIZE,
            num_active_files: int = 2,
            difference_value: int = MARGIN,
            spectrogram_nperseg: int = SPECTROGRAM_NPERSEG,
            spectrogram_frame_grouping: int = SPECTROGRAM_FRAME_GROUPING,
            ) -> None:
        super().__init__()

        if len(filenames) < 2:
            raise ValueError('minimum 2 files must be provided')
        if num_active_files < 2:
            raise ValueError('minimum 2 files must be active')
        if len(filenames) < num_active_files:
            raise ValueError('not enough files provided for num_active_files')

        self.all_filenames = filenames
        self.batch_size = batch_size
        self.dim = dim
        self.data_width = data_width
        self.num_active_files = num_active_files
        self.difference_value = difference_value
        self.spectrogram_nperseg = spectrogram_nperseg
        self.spectrogram_frame_grouping = spectrogram_frame_grouping

        self.active_spectrograms: dict[str, npt.NDArray[np.float32]] = {}
        self.active_spectrogram_queues: dict[str, list[int]] = {}
        self.active_filenames: list[str] = []

        self.curr_filename_index = 0

        self.spectrogram_lengths: dict[str, int] = {
            fname: STEPS_PER_EPOCH
            for fname in self.all_filenames
        }

        self.on_epoch_end()


    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return int((
            sum(self.spectrogram_lengths.values())
            - mean(self.spectrogram_lengths.values()) * 2
            - self.data_width * len(self.spectrogram_lengths)
        ) // self.batch_size)


    def __getitem__(
            self,
            index: int = 0,
            ) -> tuple[
                tuple[
                    npt.NDArray[np.float32],
                    npt.NDArray[np.float32],
                ],
                npt.NDArray[np.float32],
            ]:
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x1 = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        x2 = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.float32)

        # Generate data
        for i in range(self.batch_size):
            (in1, in2), out = self._get_one()

            # Store sample
            x1[i] = in1
            x2[i] = in2

            # Store class
            y[i] = out

        return (x1, x2), y

    def _get_one(self) -> Raw_Data_Point_Type:
        if random.randrange(2):
            # same
            filename = random.choice(self.active_filenames)
            idx1 = self.active_spectrogram_queues[filename].pop(0)
            idx2 = self.active_spectrogram_queues[filename].pop(0)
            sample1 = self.active_spectrograms[filename][:,idx1:idx1+self.data_width]
            sample2 = self.active_spectrograms[filename][:,idx2:idx2+self.data_width]
            result = 0
            self._delete_if_empty(filename)
        else:
            # different
            filename1, filename2 = random.sample(self.active_filenames, 2)
            idx1 = self.active_spectrogram_queues[filename1].pop(0)
            idx2 = self.active_spectrogram_queues[filename2].pop(0)
            sample1 = self.active_spectrograms[filename1][:,idx1:idx1+self.data_width]
            sample2 = self.active_spectrograms[filename2][:,idx2:idx2+self.data_width]
            result = self.difference_value
            self._delete_if_empty(filename1)
            self._delete_if_empty(filename2)

        return (sample1, sample2), result

    def _load_one_spectrogram(self) -> None:
        while filename_to_get := self.all_filenames[self.curr_filename_index]:
            if filename_to_get not in self.active_spectrograms:
                break
            self.curr_filename_index += 1
            self.curr_filename_index %= len(self.all_filenames)

        filename_to_get = self.all_filenames[self.curr_filename_index]
        # print('loading', filename_to_get)
        self.active_filenames.append(filename_to_get)
        spectrogram: npt.NDArray[np.float32]
        spectrogram = load_data(
            filename_to_get, self.spectrogram_nperseg, self.spectrogram_frame_grouping
            ).astype(np.float32)
        self.active_spectrograms[filename_to_get] = spectrogram
        self.active_spectrogram_queues[filename_to_get] = list(range(spectrogram.shape[1]-self.data_width))
        random.shuffle(self.active_spectrogram_queues[filename_to_get])
        self.active_spectrogram_queues[filename_to_get] = self.active_spectrogram_queues[filename_to_get][:1000]

        self.spectrogram_lengths[filename_to_get] = len(self.active_spectrogram_queues[filename_to_get])
        self.curr_filename_index += 1
        self.curr_filename_index %= len(self.all_filenames)

    def _delete_if_empty(self, filename: str) -> None:
        if len(self.active_spectrogram_queues[filename]) < 2:
            # print('deleting', filename)
            self.active_filenames.remove(filename)
            del self.active_spectrograms[filename]
            del self.active_spectrogram_queues[filename]

            self._load_one_spectrogram()

    def on_epoch_end(self) -> None:
        random.shuffle(self.all_filenames)
        self.curr_filename_index = 0

        self.active_spectrograms = {}
        self.active_spectrogram_queues = {}
        self.active_filenames = []

        while len(self.active_filenames) < self.num_active_files:
            self._load_one_spectrogram()

        return None



def visualize(
        pairs: npt.NDArray[np.float32],
        labels: npt.NDArray[np.float32],
        to_show: int = 6,
        num_col: int = 3,
        predictions: npt.NDArray[np.float32] | None = None,
        test: bool = False) -> Any:
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, *IMAGE_SIZE).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if predictions is not None:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()


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


def loss(margin: float = MARGIN) -> Callable:
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


def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()



print('preparing data')
input_filepaths = get_audio_files(AUDIO_FOLDER)
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
    num_active_files=15,
)
test_data = DataGenerator(
    test_filepaths,
    IMAGE_WIDTH,
    num_active_files=2,
)


"""
## Define the model

There are two input layers, each leading to its own network, which
produces embeddings. A `Lambda` layer then merges them using an
[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and the
merged output is fed to the final network.
"""


input = keras.layers.Input((*IMAGE_SIZE, 1))
x = keras.layers.BatchNormalization()(input)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = PatchScanner(6, x.shape, strides=6, layers=2, output_size=IMAGE_SIZE[0]//2)(x)
x = PatchScanner(4, x.shape, strides=2, layers=5, output_size=IMAGE_SIZE[0]//2)(x)
x = PatchScanner(4, x.shape, strides=1, layers=10, output_size=IMAGE_SIZE[0]//2)(x)
x = PatchScanner(4, x.shape, strides=2, layers=10, output_size=IMAGE_SIZE[0]//4)(x)
# x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
# x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Dense(500, activation="relu")(x)
x = keras.layers.Dense(100, activation="relu")(x)
x = keras.layers.Dense(LATENT_SPACE, activation="tanh")(x)

embedding_network = keras.Model(input, x)


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


"""
## Compile the model with the contrastive loss
"""

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

"""
## Visualize results
"""

def i_am_confusion(
        filenames: list[str],
        batch_size: int = 64,
        samples_per_song: int = 5000,
        ) -> tuple[
            dict[str, dict[str, float]],
            dict[str, dict[str, float]]]:
    """
    >>> result = i_am_confusion(filenames)
    >>> z = result[x][y]
    """
    batch_predictions: npt.NDArray[np.float32]
    batch: tuple[list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]
    confusion: dict[str, dict[str, float]]
    confidence: dict[str, dict[str, float]]

    confusion = defaultdict(dict)
    confidence = defaultdict(dict)
    for i, filename1 in enumerate(filenames):
        spectrogram1 = load_data(
            filename1, SPECTROGRAM_NPERSEG, SPECTROGRAM_FRAME_GROUPING
            ).astype(np.float32)
        for j, filename2 in enumerate(filenames[i:]):
            print(f'{100*(i+j/(len(filenames)-i))/len(filenames): >6.2f}%', end='\b'*7, flush=True)
            spectrogram2 = load_data(
                filename2, SPECTROGRAM_NPERSEG, SPECTROGRAM_FRAME_GROUPING
                ).astype(np.float32)
            preds: list[float] = []
            batch = ([], [])
            for _ in range(samples_per_song):
                idx1 = random.randrange(spectrogram1.shape[1]-IMAGE_WIDTH)
                idx2 = random.randrange(spectrogram2.shape[1]-IMAGE_WIDTH)
                image1 = spectrogram1[:,idx1:idx1+IMAGE_WIDTH]
                image2 = spectrogram2[:,idx2:idx2+IMAGE_WIDTH]
                batch[0].append(image1)
                batch[1].append(image2)
                if len(batch[0]) == batch_size:
                    batch_predictions = siamese.predict([np.array(batch[0]),np.array(batch[1])],verbose=0)
                    preds.extend(batch_predictions[:,0].tolist())
                    batch = ([], [])
            if batch[0]:
                batch_predictions = siamese.predict([np.array(batch[0]),np.array(batch[1])],verbose=0)
                preds.extend(batch_predictions[:,0].tolist())
            confusion[filename1][filename2] = mean(preds)
            confidence[filename1][filename2] = stdev(preds)
    print('100.00%')
    return confusion, confidence


def print_matrix(matrix: dict[str, dict[str, float]]) -> None:
    def fix_name(name: str) -> str:
        name = name.removeprefix('data/audio\\').removesuffix('.wav')
        name = name.replace(',', ' &')
        return name
    rows = list(matrix.keys())
    print(',' + ','.join(fix_name(r) for r in rows))
    for row in rows:
        print(f'{fix_name(row)},' + ','.join(str(matrix[row][col]) if col in matrix[row] else "" for col in rows))

a = i_am_confusion(test_filepaths[:20], batch_size=400, samples_per_song=3000)
print()
print("test confusion")
print_matrix(a[0])
print()
print()
print("test confidence")
print_matrix(a[1])
print()
print()
b = i_am_confusion(train_filepaths[:20], batch_size=400, samples_per_song=3000)
print("train confusion")
print_matrix(b[0])
print()
print()
print("train confidence")
print_matrix(b[1])
