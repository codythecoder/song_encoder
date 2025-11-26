from collections import defaultdict
import random
from typing import Optional

import numpy as np
import numpy.typing as npt



def i_am_confusion(
        encodings: dict[str, npt.NDArray[np.float32]],
        filenames: Optional[list[str]] = None,
        samples_per_song: int = 5000,
        ) -> dict[str, dict[str,  dict[str, float]]]:
    batch: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    confusion: dict[str, dict[str,  dict[str, float]]]

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


def print_matrix(matrix: dict[str, dict[str,  dict[str, float]]], key: str) -> None:
    def fix_name(name: str) -> str:
        name = name.removeprefix('data/audio\\').removesuffix('.wav')
        name = name.replace(',', ' &')
        return name

    rows = list(matrix.keys())

    print(',' + ','.join(fix_name(r) for r in rows))
    for row in rows:
        print(f'{fix_name(row)},' + ','.join(str(matrix[row][col][key]) for col in rows))
