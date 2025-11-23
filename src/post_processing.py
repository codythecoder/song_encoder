from collections import defaultdict
import random
from statistics import mean, stdev

import numpy as np
import numpy.typing as npt

from src.data import load_data
from src.encoder_types import Config_Type

import keras


def i_am_confusion(
        model: keras.Model,
        filenames: list[str],
        config: Config_Type,
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
            filename1, config['SPECTROGRAM_NPERSEG'], config['SPECTROGRAM_FRAME_GROUPING']
            ).astype(np.float32)
        for j, filename2 in enumerate(filenames[i:]):
            print(f'{100*(i+j/(len(filenames)-i))/len(filenames): >6.2f}%', end='\b'*7, flush=True)
            spectrogram2 = load_data(
                filename2, config['SPECTROGRAM_NPERSEG'], config['SPECTROGRAM_FRAME_GROUPING']
                ).astype(np.float32)
            preds: list[float] = []
            batch = ([], [])
            for _ in range(samples_per_song):
                idx1 = random.randrange(spectrogram1.shape[1]-config['IMAGE_WIDTH'])
                idx2 = random.randrange(spectrogram2.shape[1]-config['IMAGE_WIDTH'])
                image1 = spectrogram1[:,idx1:idx1+config['IMAGE_WIDTH']]
                image2 = spectrogram2[:,idx2:idx2+config['IMAGE_WIDTH']]
                batch[0].append(image1)
                batch[1].append(image2)
                if len(batch[0]) == batch_size:
                    batch_predictions = model.predict([np.array(batch[0]),np.array(batch[1])],verbose=0)
                    preds.extend(batch_predictions[:,0].tolist())
                    batch = ([], [])
            if batch[0]:
                batch_predictions = model.predict([np.array(batch[0]),np.array(batch[1])],verbose=0)
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