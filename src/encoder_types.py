from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

Image_Type: TypeAlias = npt.NDArray[np.float32]
Raw_Data_Point_Type: TypeAlias = tuple[Image_Type, Image_Type, Image_Type]

class Config_Type(TypedDict):
    AUDIO_FOLDER: str

    EPOCHS: int
    BATCH_SIZE: int
    MARGIN: int
    LATENT_SPACE: int
    TRAIN_TEST_SPLIT: float
    STEPS_PER_EPOCH: int
    SAMPLES_PER_SONG_LOAD: int

    EXPECTED_BITRATE: int
    SPECTROGRAM_NPERSEG: int
    SPECTROGRAM_FRAME_GROUPING: int

    IMAGE_WIDTH: int
    IMAGE_HEIGHT: int
    IMAGE_SIZE: tuple[int, int]
