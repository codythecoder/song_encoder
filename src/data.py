import os
import random
from statistics import mean
from typing import Optional

import numpy as np
import numpy.typing as npt

import keras

from src.encoder_types import Raw_Data_Point_Type

from src.spectrogrammify import spectrogrammify


def get_audio_files(audio_folder: str) -> list[str]:
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
        expected_bitrate: int = 44100,
        ) -> npt.NDArray[np.float64]:

    bitrate, _, _, spectrogram = spectrogrammify(filepath, nperseg=nperseg)
    if bitrate != expected_bitrate:
        print(bitrate)
        print(filepath)
        assert bitrate == expected_bitrate

    spectrogram = np.array([
        spectrogram[:,j*seg_length:j*seg_length+seg_length].max(axis=1)
        for j in range(len(spectrogram[0])//seg_length)
    ], dtype=np.float64).transpose((1, 0))

    spectrogram /= spectrogram.max()

    return spectrogram


class DataGenerator(keras.utils.PyDataset):  # type: ignore
    def __init__(
            self,
            filenames: list[str],
            data_width: int,
            batch_size: int,
            dim: tuple[int, ...],
            spectrogram_nperseg: int,
            spectrogram_frame_grouping: int,
            num_active_files: int = 2,
            margin: int = 1,
            samples_per_song_load: Optional[int] = None
            ) -> None:
        super().__init__()
        # print('__init__')

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
        self.difference_value = margin
        self.spectrogram_nperseg = spectrogram_nperseg
        self.spectrogram_frame_grouping = spectrogram_frame_grouping
        self.samples_per_song_load = samples_per_song_load

        self.active_spectrograms: dict[str, npt.NDArray[np.float32]] = {}
        self.active_spectrogram_queues: dict[str, list[int]] = {}
        self.active_filenames: list[str] = []

        self.curr_filename_index = 0

        self.spectrogram_lengths: dict[str, int] = {
            # a guess
            fname: 5000
            for fname in self.all_filenames
        }


        self.on_epoch_begin()
        assert len(self.active_filenames) >= 2
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)


    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        # print('__len__')
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)
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
                    npt.NDArray[np.float32],
                ],
                npt.NDArray[np.float32],
            ]:
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # print('__getitem__')
        # Initialization
        x1 = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        x2 = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        x3 = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        y = np.zeros((self.batch_size,2), dtype=np.float32)
        y[:,1] = self.difference_value

        # Generate data
        for i in range(self.batch_size):
            assert len(self.active_filenames) >= 2
            assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
            assert sorted(self.active_filenames) == sorted(self.active_spectrograms)
            in1, in2, in3 = self._get_one()

            # Store sample
            x1[i] = in1
            x2[i] = in2
            x3[i] = in3

        assert len(self.active_filenames) >= 2
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

        return (x1, x2, x3), y

    def _get_one(self) -> Raw_Data_Point_Type:
        # print('_get_one')
        assert len(self.active_filenames) >= 2
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

        filename1, filename2 = random.sample(self.active_filenames, 2)
        idx1 = self.active_spectrogram_queues[filename1].pop(0)
        idx2 = self.active_spectrogram_queues[filename1].pop(0)
        idx3 = self.active_spectrogram_queues[filename2].pop(0)
        sample1 = self.active_spectrograms[filename1][:,idx1:idx1+self.data_width]
        sample2 = self.active_spectrograms[filename1][:,idx2:idx2+self.data_width]
        sample3 = self.active_spectrograms[filename2][:,idx3:idx3+self.data_width]

        self._delete_if_empty(filename1)
        self._delete_if_empty(filename2)

        assert len(self.active_filenames) >= 2
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

        return sample1, sample2, sample3

    def _load_one_spectrogram(self) -> None:
        # print('_load_one_spectrogram')
        while filename_to_get := self.all_filenames[self.curr_filename_index]:
            if filename_to_get not in self.active_spectrograms:
                break
            self.curr_filename_index += 1
            self.curr_filename_index %= len(self.all_filenames)

        assert filename_to_get not in self.active_filenames
        assert filename_to_get not in self.active_spectrograms
        assert filename_to_get not in self.active_spectrogram_queues


        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

        # print('filename_to_get')
        # print(filename_to_get)
        # print('self.active_filenames')
        # print(self.active_filenames)
        # print('self.active_spectrograms')
        # print(list(self.active_spectrograms))
        # print('self.active_spectrogram_queues')
        # print(list(self.active_spectrogram_queues))


        # filename_to_get = self.all_filenames[self.curr_filename_index]
        # print('loading', filename_to_get)
        self.active_filenames.append(filename_to_get)
        spectrogram: npt.NDArray[np.float32]
        spectrogram = load_data(
            filename_to_get, self.spectrogram_nperseg, self.spectrogram_frame_grouping
            ).astype(np.float32)
        self.active_spectrograms[filename_to_get] = spectrogram
        self.active_spectrogram_queues[filename_to_get] = list(range(spectrogram.shape[1]-self.data_width))
        random.shuffle(self.active_spectrogram_queues[filename_to_get])

        if self.samples_per_song_load is not None:
            self.active_spectrogram_queues[filename_to_get] = self.active_spectrogram_queues[filename_to_get][:self.samples_per_song_load]

        self.spectrogram_lengths[filename_to_get] = len(self.active_spectrogram_queues[filename_to_get])
        self.curr_filename_index += 1
        self.curr_filename_index %= len(self.all_filenames)

        # assert self.active_filenames
        # assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        # assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

    def _delete_if_empty(self, filename: str) -> None:
        # print('_delete_if_empty')
        if len(self.active_spectrogram_queues[filename]) < 2:
            # print('deleting', filename)
            self.active_filenames.remove(filename)
            del self.active_spectrograms[filename]
            del self.active_spectrogram_queues[filename]

            assert self.active_filenames
            assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
            assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

            self._load_one_spectrogram()


        assert len(self.active_filenames) >= 2
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

    def on_epoch_begin(self) -> None:
        # print('on_epoch_end')
        random.shuffle(self.all_filenames)
        self.curr_filename_index = 0

        self.active_spectrograms = {}
        self.active_spectrogram_queues = {}
        self.active_filenames = []

        while len(self.active_filenames) < self.num_active_files:
            self._load_one_spectrogram()


        assert len(self.active_filenames) >= 2
        assert sorted(self.active_filenames) == sorted(self.active_spectrogram_queues)
        assert sorted(self.active_filenames) == sorted(self.active_spectrograms)

        return None
