from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

Image_Type: TypeAlias = npt.NDArray[np.float32]
Raw_Data_Point_Type: TypeAlias = tuple[Image_Type, Image_Type, Image_Type]


__Similarity_Data = TypedDict(
    '__Similarity_Data',
    {
        'dist_mean': float,
        'avg_from_mean': float,
        # 'std_from_mean': float,
    }
)
Similarity_Matrix = dict[str, dict[str, __Similarity_Data]]


class Config(TypedDict):
    LATENT_SPACE: int
    "the feature space from the output of the model"

    EXPECTED_BITRATE: int
    """
    the bitrate that will be expected from the audio files
        higher bitrates will be downsampled"""

    SPECTROGRAM_NPERSEG: int
    "takes n samples from the audiofile to calculate the spectrogram"

    SPECTROGRAM_FRAME_GROUPING: int
    "takes n frames from the spectrogram to combine"

    IMAGE_WIDTH: int
    "how many frames to input to the learning model"

    IMAGE_HEIGHT: int
    "how many frequency buckets in the spectrogram"

