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
from typing import Callable, Optional, Any

import keras
from keras import ops
from keras import layers

import numpy as np
from numpy import typing as npt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from spectrogrammify import spectrogrammify


#
# Hyperparameters
#

EPOCHS = 10
BATCH_SIZE = 16
MARGIN = 1  # Margin for contrastive loss.
LATENT_SPACE = 10
# IMAGE_SIZE = 28, 28



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
        nperseg: int = 256,
        seg_length: int = 1500,
        data_length: int = 12,
        stride: int = 1,
        ) -> tuple[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    seg_length = (seg_length // nperseg)

    folderpath = r'data/audio'
    filenames = os.listdir(folderpath)

    train_x: list[npt.NDArray[np.float64]] = []
    test_x: list[npt.NDArray[np.float64]] = []
    train_y: list[int] = []
    test_y: list[int] = []
    for i, filename in enumerate(filenames):
        bitrate, _, _, spectrogram = spectrogrammify(os.path.join(folderpath, filename), nperseg=nperseg)
        data_len = ((data_length*bitrate) // (seg_length*nperseg))
        spectrogram = np.array([
            spectrogram[:,j*seg_length:j*seg_length+seg_length].mean(axis=1)
            for j in range(len(spectrogram[0])//seg_length)
        ], dtype=np.float64).transpose((1, 0))
        train_cutoff: int = int((spectrogram.shape[1] - data_len) * 0.8)
        train_x.extend(
            spectrogram[:,j:j+data_len]
            for j in range(0, train_cutoff, stride)
        )
        test_x.extend(
            spectrogram[:,j:j+data_len]
            for j in range(train_cutoff, spectrogram.shape[1] - data_len, stride)
        )
        train_y.extend([i] * len(range(0, train_cutoff, stride)))
        test_y.extend([i] * len(range(train_cutoff, spectrogram.shape[1] - data_len, stride)))

    # random.shuffle(pairs)
    # split_index = int(len(pairs)*0.8)
    # train_x = np.array([pairs[i][0] for i in range(split_index)])
    # train_y = np.array([pairs[i][1] for i in range(split_index)])
    # pairs = pairs[split_index:]
    return (
        (
            np.array(train_x),
            np.array(train_y),
        ),
        (
            np.array(test_x),
            np.array(test_y),
        ),
    )



#
# Create pairs of images
#
# We will train the model to differentiate between digits of different classes. For
# example, digit `0` needs to be differentiated from the rest of the
# digits (`1` through `9`), digit `1` - from `0` and `2` through `9`, and so on.
# To carry this out, we will select N random images from class A (for example,
# for digit `0`) and pair them with N random images from another class B
# (for example, for digit `1`). Then, we can repeat this process for all classes
# of digits (until digit `9`). Once we have paired digit `0` with other digits,
# we can repeat this process for the remaining classes for the rest of the digits
# (from `1` until `9`).
#


def make_pairs(
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32]
        ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0].tolist() for i in range(num_classes)]

    pairs: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs.append((x1, x2))
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs.append((x1, x2))
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


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



print('loading dataset')
#
# Load the MNIST dataset
#
# (x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()
(_x_train, _y_train), (_x_test, _y_test) = load_data(nperseg = 128, stride=2)
print('formatting data')

# Change the data type to a floating point format
x_train: npt.NDArray[np.float32] = _x_train.astype("float32")
x_test: npt.NDArray[np.float32] = _x_test.astype("float32")
y_train: npt.NDArray[np.int8] = _y_train.astype("int8")
y_test: npt.NDArray[np.int8] = _y_test.astype("int8")


#
# Define training and validation sets
#

# Keep 50% of train_val  in validation set
# x_train, x_val = x_train_val[:30000], x_train_val[30000:]
# y_train, y_val = y_train_val[:30000], y_train_val[30000:]
# del x_train_val, y_train_val




# print('testing patches')

# PATCH_SHAPE = 36, 9
# IMAGE_SIZE = 36, 36
# BATCH_SIZE = 2
# COLOR_CHANNELS = 3

# images = [
#     Image.open(r'temp1.png'),
#     Image.open(r'temp2.png'),
# ]

# first_array=np.reshape(images[:BATCH_SIZE], (BATCH_SIZE, *IMAGE_SIZE, 3))
# first_array = first_array.astype("float32")/255

# patches = keras.ops.image.extract_patches(first_array, PATCH_SHAPE, padding='valid')

# num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]
# patch_dim = keras.ops.shape(patches)[3]

# out = keras.ops.reshape(patches, (BATCH_SIZE, num_patches, *PATCH_SHAPE, COLOR_CHANNELS))
# # out = keras.ops.transpose(out, axes=(0, 2, 3, 4, 1))

# f, axarr = plt.subplots(1,4)

# # axarr[0].imshow(out[0,:,:,:,0])
# # axarr[1].imshow(out[0,:,:,:,1])
# # axarr[2].imshow(out[0,:,:,:,2])
# # axarr[3].imshow(out[0,:,:,:,3])
# axarr[0].imshow(out[0,0])
# axarr[1].imshow(out[0,1])
# axarr[2].imshow(out[0,2])
# axarr[3].imshow(out[0,3])

# #Actually displaying the plot if you are not in interactive mode
# plt.show()
# # input()






# make train pairs
pairs_train, labels_train = make_pairs(x_train, y_train)

# make validation pairs
# pairs_val, labels_val = make_pairs(x_val, y_val)

# make test pairs
pairs_test, labels_test = make_pairs(x_test, y_test)

"""
We get:

**pairs_train.shape = (60000, 2, *IMAGE_SIZE)**

- We have 60,000 pairs
- Each pair contains 2 images
- Each image has shape `(*IMAGE_SIZE)`
"""

"""
Split the training pairs
"""

x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, *IMAGE_SIZE)
x_train_2 = pairs_train[:, 1]

"""
Split the validation pairs
"""

# x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, *IMAGE_SIZE)
# x_val_2 = pairs_val[:, 1]

"""
Split the test pairs
"""

x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, *IMAGE_SIZE)
x_test_2 = pairs_test[:, 1]


"""
## Visualize pairs and their labels
"""


"""
Inspect training pairs
"""

# visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)

"""
Inspect validation pairs
"""

# visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=4)

"""
Inspect test pairs
"""

# visualize(pairs_test[:-1], labels_test[:-1], to_show=4, num_col=4)

"""
## Define the model

There are two input layers, each leading to its own network, which
produces embeddings. A `Lambda` layer then merges them using an
[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and the
merged output is fed to the final network.
"""


IMAGE_SIZE = x_test_1.shape[1:]
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
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_test_1, x_test_2], labels_test),
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