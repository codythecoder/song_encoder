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
from typing import Callable, Generator, Iterable, Optional, Any, TypeAlias

import keras
from keras import ops
from keras import layers

import numpy as np
from numpy import typing as npt

import matplotlib.pyplot as plt

from spectrogrammify import spectrogrammify

Image_Type: TypeAlias = npt.NDArray[np.float32]
Raw_Data_Point_Type: TypeAlias = tuple[tuple[Image_Type, Image_Type], int]


#
# Hyperparameters
#

EPOCHS = 10
BATCH_SIZE = 16
MARGIN = 1  # Margin for contrastive loss.
LATENT_SPACE = 10
TRAIN_TEST_SPLIT = 0.8
MAX_BATCHES_PER_EPOCH = 5000

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
        dense = keras.layers.Dense(self.output_height*self.layers, activation='relu')(dense)
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


def load_data(
        filepath: str,
        nperseg: int = 256,
        seg_length: int = 1,
        ) -> npt.NDArray[np.float64]:

    bitrate, _, _, spectrogram = spectrogrammify(filepath, nperseg=nperseg)
    assert bitrate == EXPECTED_BITRATE

    spectrogram = np.array([
        spectrogram[:,j*seg_length:j*seg_length+seg_length].mean(axis=1)
        for j in range(len(spectrogram[0])//seg_length)
    ], dtype=np.float64).transpose((1, 0))

    return spectrogram

# a = load_data('data/audio/hsp2-v5.wav', SPECTROGRAM_NPERSEG, SPECTROGRAM_FRAME_GROUPING)

def stream_pairs(
        filenames: list[str],
        data_width: int,
        num_active_files: int = 2,
        difference_value: int = 1,
        spectrogram_nperseg: int = SPECTROGRAM_NPERSEG,
        spectrogram_frame_grouping: int = SPECTROGRAM_FRAME_GROUPING,
        ) -> Callable[..., Generator[Raw_Data_Point_Type, None, None]]:
    if num_active_files < 2:
        raise ValueError('`num_active_files` must be 2 or greater')
    if num_active_files >= len(filenames):
        raise ValueError('`num_active_files` must be less than the number of filenames')

    active_spectrograms: dict[str, npt.NDArray[np.float32]] = {}
    active_spectrogram_queues: dict[str, list[int]] = {}
    active_filenames: list[str] = []

    def pairs() -> Generator[Raw_Data_Point_Type, None, None]:
        def delete_if_empty(filename: str) -> None:
            if len(active_spectrogram_queues[filename]) < 2:
                # print('deleting', filename)
                active_filenames.remove(filename)
                del active_spectrograms[filename]
                del active_spectrogram_queues[filename]

        while True:
            while len(active_filenames) < num_active_files:
                filename_to_get = random.choice([f for f in filenames if f not in active_spectrograms])
                # print('loading', filename_to_get)
                active_filenames.append(filename_to_get)
                spectrogram: npt.NDArray[np.float32]
                spectrogram = load_data(
                    filename_to_get, spectrogram_nperseg, spectrogram_frame_grouping
                    ).astype(np.float32)
                active_spectrograms[filename_to_get] = spectrogram
                active_spectrogram_queues[filename_to_get] = list(range(spectrogram.shape[1]-data_width))
                random.shuffle(active_spectrogram_queues[filename_to_get])

            if random.randrange(2):
                # same
                filename = random.choice(active_filenames)
                idx1 = active_spectrogram_queues[filename].pop(0)
                idx2 = active_spectrogram_queues[filename].pop(0)
                sample1 = active_spectrograms[filename][:,idx1:idx1+data_width]
                sample2 = active_spectrograms[filename][:,idx2:idx2+data_width]
                result = 0
                delete_if_empty(filename)
            else:
                # different
                filename1, filename2 = random.sample(active_filenames, 2)
                idx1 = active_spectrogram_queues[filename1].pop(0)
                idx2 = active_spectrogram_queues[filename2].pop(0)
                sample1 = active_spectrograms[filename1][:,idx1:idx1+data_width]
                sample2 = active_spectrograms[filename2][:,idx2:idx2+data_width]
                result = difference_value
                delete_if_empty(filename1)
                delete_if_empty(filename2)

            yield (sample1, sample2), result

    return pairs


class DataGenerator(keras.utils.PyDataset):
    def __init__(
            self,
            streamer: Callable[..., Iterable[Raw_Data_Point_Type]],
            dim: tuple[int, ...] = IMAGE_SIZE,
            ) -> None:
        super().__init__()
        self.streamer = streamer
        self.dim = dim

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return MAX_BATCHES_PER_EPOCH

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
        x1 = np.empty((BATCH_SIZE, *self.dim), dtype=np.float32)
        x2 = np.empty((BATCH_SIZE, *self.dim), dtype=np.float32)
        y = np.empty((BATCH_SIZE,), dtype=np.float32)

        # Generate data
        for i, ((in1, in2), out) in zip(range(BATCH_SIZE), self.streamer()):
            # Store sample
            x1[i] = in1
            x2[i] = in2

            # Store class
            y[i] = out

        return (x1, x2), y



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


def loss(margin: float = 1) -> Callable:
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




input_filepaths = [os.path.join(AUDIO_FOLDER, fname) for fname in os.listdir(AUDIO_FOLDER)]
train_filepaths = input_filepaths[:int(len(input_filepaths)*TRAIN_TEST_SPLIT)]
test_filepaths = [fpath for fpath in input_filepaths if fpath not in train_filepaths]

train_streamer = stream_pairs(train_filepaths, IMAGE_WIDTH, num_active_files=5)
test_streamer = stream_pairs(test_filepaths, IMAGE_WIDTH, num_active_files=2)

train_data = DataGenerator(train_streamer)
test_data = DataGenerator(test_streamer)


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

siamese.compile(loss=loss(margin=MARGIN), optimizer="RMSprop", metrics=["accuracy"])
embedding_network.summary()
siamese.summary()


"""
## Train the model
"""

history = siamese.fit(
    x=train_data,
    validation_data=test_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)

siamese.save('out.keras')

"""
## Visualize results
"""


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the contrastive loss
plt_metric(history=history.history, metric="loss", title="Contrastive Loss")

"""
## Evaluate the model
"""

results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)

"""
## Visualize the predictions
"""

predictions = siamese.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)