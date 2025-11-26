from collections import defaultdict
import random
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

import keras

from src.data import load_data

from src.encoder_types import Config, Similarity_Matrix



def calculate_similarity_matrix(
        encodings: dict[str, npt.NDArray[np.float32]],
        filenames: Optional[list[str]] = None,
        samples_per_song: int = 5000,
        ) -> Similarity_Matrix:

    batch: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    confusion: Similarity_Matrix

    if filenames is None:
        filenames = list(encodings.keys())

    confusion = defaultdict(dict)
    for fname_i, filename1 in enumerate(filenames):
        for fname_j, filename2 in enumerate(filenames):
            print(f'{100*(fname_i+fname_j/len(filenames))/len(filenames): >6.2f}%', end='\b'*7, flush=True)

            batch = (
                np.empty((samples_per_song, encodings[filename1].shape[1]), dtype=np.float32),
                np.empty((samples_per_song, encodings[filename1].shape[1]), dtype=np.float32),
            )

            for i in range(samples_per_song):
                idx1 = random.randrange(encodings[filename1].shape[0])
                idx2 = random.randrange(encodings[filename2].shape[0])
                image1 = encodings[filename1][idx1]
                image2 = encodings[filename2][idx2]
                batch[0][i] = image1
                batch[1][i] = image2

            confusion[filename1][filename2] = {
                'dist_mean': float(np.linalg.norm(batch[0].mean(0) - batch[1].mean(0))),
                'avg_from_mean': float(np.linalg.norm(batch[1] - batch[0].mean(0), axis=1).mean()),
            }

    print('100.00%')
    return confusion


def __fix_name(name: str) -> str:
    # get just the filename
    name = name.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    # remove filetype
    name = name.rsplit('.', 1)[-1]
    name = name.replace(',', ' &')
    return name


def print_matrix(matrix: Similarity_Matrix, key: Literal['dist_mean'] | Literal['avg_from_mean']) -> None:
    rows = list(matrix.keys())

    print(',' + ','.join(__fix_name(r) for r in rows))
    for row in rows:
        print(f'{__fix_name(row)},' + ','.join(str(matrix[row][col][key]) for col in rows))


def get_song_encodings(embedded_model: keras.Model, filename: str, config: Config, batch_size: int = 64) -> npt.NDArray[np.float32]:
    spectrogram = load_data(filename, config).astype(np.float32)

    end_idx = spectrogram.shape[1]-config['IMAGE_WIDTH']
    preds = np.empty((end_idx, config['LATENT_SPACE']), dtype=np.float32)

    for idx in range(0, end_idx, batch_size):
        curr_end_idx = min(idx+batch_size, end_idx)
        batch = [
            spectrogram[:, i:i+config['IMAGE_WIDTH']]
            for i in range(idx, curr_end_idx)
        ]

        new_preds = embedded_model.predict(
            np.array(batch),
            verbose=0,
        )

        preds[idx:curr_end_idx] = new_preds

    return preds

