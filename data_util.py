""""
Creates dataloaders for the RAVDESS dataset so that it is compatible with the augmentation strategy
"""""
import tensorflow as tf
from pathlib import Path
import scipy
import numpy as np
import librosa

from config import TESS_ORIGINAL_FOLDER_PATH


def mel_to_mfcc(mel, num_mfcc=40, dct_type=2):
    if not isinstance(mel, np.ndarray):
        mel = mel.cpu().numpy()
    mfccs = scipy.fftpack.dct(mel, axis=-2, type=dct_type, norm='ortho')[..., :num_mfcc, :]
    return np.mean(mfccs.T, axis=0)


class RavdessDataset:
    def __init__(self,  batch_sz, dataset_root_dir: Path, augmenter=None):
        self.val_ds = None
        self.train_ds = None
        self.loaded_dataset = None
        self.batch_sz = batch_sz
        self.size = 0
        self.sampling_rate = 0
        self.dataset = tf.data.Dataset.list_files(f"{dataset_root_dir}/*.wav")
        if augmenter is None:
            self.augmenter = None
        else:
            self.augmenter = augmenter.get(self.mel_fn)

        self.load_process()
        self.split(val_pct=0.33)

    def mel_fn(self, signal):
        mel_power = librosa.feature.melspectrogram(y=signal, sr=self.sampling_rate)
        mel_db = librosa.core.power_to_db(mel_power)
        return mel_db

    def get_class(self, filepath: Path) -> int:
        label = int(filepath.name[7:8]) - 1
        return label

    def load_audio_features(self, filepath: Path) -> tf.Tensor:
        X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
        self.sampling_rate = sample_rate
        # mfccs = self.mel_fn(X)
        return X

    def load_labelled_datapoint(self, filepath: tf.Tensor):
        filepath = Path(bytes.decode(filepath.numpy()))
        label = self.get_class(filepath)
        audio = self.load_audio_features(filepath)
        return audio, label

    def load_process(self, shuffle_size=1000):
        signals = []
        labels = []
        for filepath in self.dataset:
            f, l = self.load_labelled_datapoint(filepath)
            signals.append(f)
            labels.append(l)
            self.size += 1

        signals = tf.keras.preprocessing.sequence.pad_sequences(signals)
        self.loaded_dataset = tf.data.Dataset.from_tensor_slices((signals, labels))

    def split(self, val_pct):
        train_pct = 1 - val_pct
        train_size = int(train_pct * self.size)
        val_size = int(val_pct * self.size)
        train_ds = self.loaded_dataset.take(train_size)
        val_ds = self.loaded_dataset.skip(train_size).take(val_size)
        # Shuffle data and create batches
        train_ds = train_ds.shuffle(buffer_size=1000)
        self.train_ds = train_ds.batch(self.batch_sz)

        # Make dataset fetch batches in the background during the training of the model.

        val_ds = val_ds.shuffle(buffer_size=1000)
        self.val_ds = val_ds.batch(self.batch_sz)


    def get_train_batch(self):
        # only augment train batch
        batch = next(iter(self.train_ds))
        signals, labels = batch
        signals = signals.numpy().astype(np.float)
        if self.augmenter is not None:
            aug_batch = self.augmenter.augment_batch(list(zip(labels, signals)))
            labels, aug_melspecs = list(zip(*aug_batch))
            aug_mfccs = [mel_to_mfcc(m) for m in aug_melspecs]
            batch = (tf.convert_to_tensor(aug_mfccs), labels)
        else:
            mfccs = []
            for signal in signals:
                signal = signal.numpy().astype(np.float)
                m = mel_to_mfcc(self.mel_fn(signal))
                mfccs.append(m)
            batch = (tf.convert_to_tensor(mfccs), labels)
        return batch

    def get_val_batch(self):
        signals, labels = next(iter(self.val_ds))
        mfccs = []
        for signal in signals:
            signal = signal.numpy().astype(np.float)
            m = mel_to_mfcc(self.mel_fn(signal))
            mfccs.append(m)
        batch = (tf.convert_to_tensor(mfccs), labels)
        return batch


if __name__ == "__main__":
    from audio_aug.augment import get_augment_schemes
    import logging

    template = "audio_aug/specaugment_scheme.yml"
    runs = get_augment_schemes(gammas=[0.5, 0.75, 0.875, 1],
                               num_runs=1,
                               template_file=template,
                               name_prefix="something")

    for run_num, augmentor in enumerate(runs):
        logging.info(f"Starting Run: {augmentor.config['run_name']} with gamma={augmentor.config['params']['gamma']}")
        rd = RavdessDataset(16, TESS_ORIGINAL_FOLDER_PATH, augmenter=augmentor)
        for batch_num, item in enumerate(rd.train_ds):
            print("-------------------------------")
            print(f"batch_num: {batch_num}, item: {item[0].shape}")
            mfccs, labels = rd.get_val_batch()
            print(mfccs.shape, labels.shape)
            print("-------------------------------")
