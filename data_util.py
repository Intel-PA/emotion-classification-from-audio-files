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
            assert sample_rate == SAMPLING_RATE, f"Wrong sampling rate {sample_rate} for {filepath}"

            self.data.append((Path(filepath), X))

        # self.data = [
        #     (
        #         fpath,
        #         signal,
        #         np.pad(signal,
        #                (0, self.longest_signal - len(signal)),
        #                constant_values=0)
        #     )
        #     for signal, fpath in self.data
        # ]


class CollateFn(object):
    @staticmethod
    def get_class(filepath: Path) -> int:
        label = int(filepath.name[7:8]) - 1
        return label

    def __init__(self, batch_sz, mel_fn=None, augmentor=None):
        self.batch_sz = batch_sz
        self.mel_fn = mel_fn
        if augmentor is not None:
            self.augmentor = augmentor.get(self.mel_fn)
        else:
            self.augmentor = None

    def __call__(self, batch):
        if self.augmentor is not None:
            # augment the batch
            batch = self.augmentor.augment_batch(batch)
        else:
            batch = [(filepath, self.mel_fn(signal)) for filepath, signal in batch]
        mfccs = []
        labels = []
        mfccs_batch_tensor = torch.cuda.FloatTensor(self.batch_sz, 40)
        for filepath, mel in batch:
            mfcc = torch.from_numpy(mel_to_mfcc(mel)).reshape(1, 40).to(DEVICE)
            mfccs.append(mfcc)
            label = CollateFn.get_class(filepath)
            labels.append(label)

        torch.cat(mfccs, out=mfccs_batch_tensor)
        label_batch_tensor = torch.LongTensor(labels)
        return mfccs_batch_tensor, label_batch_tensor


def load_data(data_path, augmentor, batch_sz=100, train_val_test_split=[0.67, 0.33, 0]):
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
                          collate_fn=CollateFn(batch_sz, mel_fn=mel_fn, augmentor=augmentor))

    val_dl = DataLoader(val_split,
                        drop_last=True,
                        batch_size=batch_sz,
                        shuffle=True,
                        collate_fn=CollateFn(batch_sz, mel_fn=mel_fn))

    return train_dl, val_dl


if __name__ == "__main__":
    from audio_aug.augment import get_augment_schemes
    import logging

    template = "audio_aug/adsmote_scheme.yml"
    runs = get_augment_schemes(gammas=[0.5, 0.75, 0.875, 1],
                               num_runs=1,
                               template_file=template,
                               name_prefix="something")

    for run_num, augmentor in enumerate(runs):
        logging.info(f"Starting Run: {augmentor.config['run_name']} with gamma={augmentor.config['params']['gamma']}")
        train_dl, val_dl = load_data(TESS_ORIGINAL_FOLDER_PATH, augmentor, batch_sz=16)

        for batch_num, (mfccs_batch, label_batch) in enumerate(train_dl):
            print("-------------------------------")
            print(f"{batch_num}: {mfccs_batch}, {label_batch.shape}")
            print("-------------------------------")
