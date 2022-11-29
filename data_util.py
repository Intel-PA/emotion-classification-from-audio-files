""""
Creates dataloaders for the RAVDESS dataset so that it is compatible with the augmentation strategy
"""""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from pathlib import Path
import scipy
import numpy as np
import librosa

from config import TESS_ORIGINAL_FOLDER_PATH


sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
cudnn_version = sys_details["cudnn_version"]
print(f"CUDA: {cuda_version}, CUDNN: {cudnn_version}")


def mel_to_mfcc(mel, num_mfcc=40, dct_type=2):
    mfccs = scipy.fftpack.dct(mel, axis=-2, type=dct_type, norm='ortho')[..., :num_mfcc, :]
    return np.mean(mfccs.T, axis=0)


def get_class(filepath: Path) -> int:
    label = int(filepath.name[7:8]) - 1
    return label


def load_audio_features(filepath: Path) -> tf.Tensor:
    X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
    mel_power = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    mel_db = librosa.core.power_to_db(mel_power)
    mfccs = mel_to_mfcc(mel_db)
    return tf.convert_to_tensor(mfccs)


def load_labelled_datapoint(filepath: tf.Tensor):
    filepath = Path(bytes.decode(filepath.numpy()))
    label = get_class(filepath)
    audio = load_audio_features(filepath)
    return audio, label


class RavdessDataset:
    def __init__(self, batch_sz, dataset_root_dir: Path):
        self.loaded_dataset = None
        self.batch_sz = batch_sz
        self.size = 0
        self.dataset = tf.data.Dataset.list_files(f"{dataset_root_dir}/*.wav")

    def load_process(self, shuffle_size=1000):
        features = []
        labels = []
        for filepath in self.dataset:
            f, l = load_labelled_datapoint(filepath)
            features.append(f)
            labels.append(l)
            self.size += 1

        self.loaded_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        # Shuffle data and create batches
        self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)
        self.loaded_dataset = self.loaded_dataset.repeat()
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_sz)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def split(self, val_pct):
        train_pct = 1 - val_pct
        train_size = int(train_pct * self.size)
        val_size = int(val_pct * self.size)
        train_ds = self.loaded_dataset.take(train_size)
        val_ds = self.loaded_dataset.skip(train_size).take(val_size)
        return train_ds, val_ds

    def get_batch(self):
        return next(iter(self.loaded_dataset))





if __name__ == "__main__":
    rd = RavdessDataset(16, TESS_ORIGINAL_FOLDER_PATH)
    rd.load_process()
    t, v = rd.split(val_pct=0.33)
    for a, l in v:
        print(a.shape, l.shape)


