""""
Creates dataloaders for the RAVDESS dataset so that it is compatible with the augmentation strategy
"""""
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path
import scipy
import numpy as np
import librosa

from config import TESS_ORIGINAL_FOLDER_PATH

SAMPLING_RATE = 22050  # TODO: move this to hparams file
DEVICE = torch.device("cuda")


def mel_to_mfcc(mel, num_mfcc=40, dct_type=2):
    if not isinstance(mel, np.ndarray):
        mel = mel.cpu().numpy()
    mfccs = scipy.fftpack.dct(mel, axis=-2, type=dct_type, norm='ortho')[..., :num_mfcc, :]
    return np.mean(mfccs.T, axis=0)


def mel_fn(signal):
    if not isinstance(signal, np.ndarray):
        signal = signal.squeeze(0)
        signal = signal.cpu().numpy()
    mel_power = librosa.feature.melspectrogram(y=signal, sr=SAMPLING_RATE)
    mel_db = librosa.core.power_to_db(mel_power)
    return mel_db


def batch_mel_to_mfcc(mel_batch):
    mfccs = []
    for mel in mel_batch:
        mfccs.append(torch.from_numpy(mel_to_mfcc(mel)).unsqueeze(0).to(DEVICE))
    return mfccs


def batch_mel_fn(signal_batch):
    mels = []
    for mel in signal_batch:
        mels.append(mel_fn(mel))
    return mels


# def unpad(tensor):
#     tensor = tf.expand_dims(tensor, 0)
#     ragged = tf.RaggedTensor.from_tensor(tensor, padding=-1447.0)
#     return tf.squeeze(ragged.to_tensor(), axis=0).numpy()


class RAVDESSDataset(Dataset):

    def __init__(self, data_root_dir: str):
        super().__init__()
        self.longest_signal = 0
        self.data = []
        self.load_data(data_root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def load_data(self, data_root_dir):
        fs = Path(data_root_dir).glob('**/*.wav')
        filepaths = [f.resolve() for f in fs]

        for filepath in filepaths:
            X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
            if X.shape[0] > self.longest_signal:
                self.longest_signal = X.shape[0]

            assert sample_rate == SAMPLING_RATE, f"Wrong sampling rate {sample_rate} for {filepath}"

            self.data.append((X, Path(filepath)))

        self.data = [
            (
                np.pad(signal,
                       (0, self.longest_signal - len(signal)),
                       constant_values=0),
                fpath
            )
            for signal, fpath in self.data
        ]


class CollateFn(object):
    @staticmethod
    def get_class(filepath: Path) -> int:
        label = int(filepath.name[7:8]) - 1
        return label

    def __init__(self, max_signal_len, batch_sz):
        self.max_signal_len = max_signal_len
        self.batch_sz = batch_sz

    def __call__(self, batch):
        signal_batch_tensor = torch.FloatTensor(self.batch_sz, self.max_signal_len)
        signals = []
        labels = []
        for signal, filepath in batch:
            signal = torch.from_numpy(signal).reshape(1, self.max_signal_len)
            signals.append(signal)
            label = CollateFn.get_class(filepath)
            labels.append(label)

        torch.cat(signals, out=signal_batch_tensor)
        label_batch_tensor = torch.LongTensor(labels)
        return signal_batch_tensor, label_batch_tensor


def load_data(data_path, batch_sz=100, train_val_test_split=[0.67, 0.33, 0]):
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"
    dataset = RAVDESSDataset(data_path)

    tr_va_te = []
    for frac in train_val_test_split:
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_va_te.append(actual_count)

    train_split, val_split, test_split = random_split(dataset, tr_va_te)

    train_dl = DataLoader(train_split,
                          drop_last=True,
                          batch_size=batch_sz,
                          shuffle=True,
                          collate_fn=CollateFn(dataset.longest_signal, batch_sz))
    val_dl = DataLoader(val_split,
                        drop_last=True,
                        batch_size=batch_sz,
                        shuffle=True,
                        collate_fn=CollateFn(dataset.longest_signal, batch_sz))

    return train_dl, val_dl


# class RavdessDataset:
#     def __init__(self,  batch_sz, dataset_root_dir: Path, augmenter=None):
#         self.val_ds = None
#         self.train_ds = None
#         self.loaded_dataset = None
#         self.batch_sz = batch_sz
#         self.size = 0
#         self.sampling_rate = 0
#         self.dataset = tf.data.Dataset.list_files(f"{dataset_root_dir}/*.wav")
#         if augmenter is None:
#             self.augmenter = None
#         else:
#             self.augmenter = augmenter.get(self.mel_fn)
#
#         self.load_process()
#         self.split(val_pct=0.33)
#
#     def mel_fn(self, signal):
#         mel_power = librosa.feature.melspectrogram(y=signal, sr=self.sampling_rate)
#         mel_db = librosa.core.power_to_db(mel_power)
#         return mel_db
#
#     def get_class(self, filepath: Path) -> int:
#         label = int(filepath.name[7:8]) - 1
#         return label
#
#     def load_audio_features(self, filepath: Path) -> tf.Tensor:
#         X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
#         self.sampling_rate = sample_rate
#         # mfccs = self.mel_fn(X)
#         return X
#
#     def load_labelled_datapoint(self, filepath: tf.Tensor):
#         filepath = Path(bytes.decode(filepath.numpy()))
#         label = self.get_class(filepath)
#         audio = self.load_audio_features(filepath)
#         return audio, label
#
#     def load_process(self, shuffle_size=1000):
#         signals = []
#         labels = []
#         for filepath in self.dataset:
#             f, l = self.load_labelled_datapoint(filepath)
#             signals.append(f)
#             labels.append(l)
#             self.size += 1
#
#         signals = tf.keras.preprocessing.sequence.pad_sequences(signals, value=-1447, padding='post', dtype=np.float)
#         self.loaded_dataset = tf.data.Dataset.from_tensor_slices((signals, labels))
#
#     def split(self, val_pct):
#         train_pct = 1 - val_pct
#         train_size = int(train_pct * self.size)
#         val_size = int(val_pct * self.size)
#         train_ds = self.loaded_dataset.take(train_size)
#         val_ds = self.loaded_dataset.skip(train_size).take(val_size)
#         # Shuffle data and create batches
#         train_ds = train_ds.shuffle(buffer_size=1000)
#         self.train_ds = train_ds.batch(self.batch_sz)
#
#         # Make dataset fetch batches in the background during the training of the model.
#
#         val_ds = val_ds.shuffle(buffer_size=1000)
#         self.val_ds = val_ds.batch(self.batch_sz)
#
#
#     def get_train_batch(self):
#         # only augment train batch
#         batch = next(iter(self.train_ds))
#         signals, labels = batch
#         signals = signals.numpy().astype(np.float)
#         signals = [unpad(s) for s in signals]
#         if self.augmenter is not None and self.augmenter.gamma < 1.0:
#             aug_batch = self.augmenter.augment_batch(list(zip(labels, signals)))
#             labels, aug_melspecs = list(zip(*aug_batch))
#             aug_mfccs = [mel_to_mfcc(m) for m in aug_melspecs]
#             batch = (tf.convert_to_tensor(aug_mfccs), labels)
#         else:
#             mfccs = []
#             for signal in signals:
#                 signal = signal.astype(np.float)
#                 m = mel_to_mfcc(self.mel_fn(signal))
#                 mfccs.append(m)
#             batch = (tf.convert_to_tensor(mfccs), labels)
#         return batch
#
#     def get_val_batch(self):
#         signals, labels = next(iter(self.val_ds))
#         signals = [unpad(s) for s in signals]
#         mfccs = []
#         for signal in signals:
#             signal = signal.astype(np.float)
#             m = mel_to_mfcc(self.mel_fn(signal))
#             mfccs.append(m)
#         batch = (tf.convert_to_tensor(mfccs), labels)
#         return batch


if __name__ == "__main__":
    # from audio_aug.augment import get_augment_schemes
    # import logging
    #
    # template = "audio_aug/specaugment_scheme.yml"
    # runs = get_augment_schemes(gammas=[0.5, 0.75, 0.875, 1],
    #                            num_runs=1,
    #                            template_file=template,
    #                            name_prefix="something")
    #
    # for run_num, augmentor in enumerate(runs):
    #     logging.info(f"Starting Run: {augmentor.config['run_name']} with gamma={augmentor.config['params']['gamma']}")
    #     train_dl, val_dl = load_data(TESS_ORIGINAL_FOLDER_PATH)
    #
    #     for batch_num, item in enumerate(val_dl):
    #         print("-------------------------------")
    #         print(f"batch_num: {batch_num}, item: {item[0].shape}")
    #         print("-------------------------------")
    train_ds, val_ds = load_data(TESS_ORIGINAL_FOLDER_PATH, 16)
    for batch_num, (signal_batch, label_batch) in enumerate(val_ds):
        print(f"{batch_num}: {signal_batch.shape}, {label_batch.shape}")
