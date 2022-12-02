"""
This files creates the X and y features in joblib to be used by the predictive models.
"""
import math
import os
import random
import sys
import time
import joblib
import librosa
import numpy
import numpy as np
from pathlib import Path

import scipy
import torch
from config import SAVE_DIR_PATH
from config import TRAINING_FILES_PATH
from specaugment.specaugment import time_mask, freq_mask, time_warp


class CreateFeatures:
    @staticmethod
    def mel_to_mfcc(mel, num_mfcc=40, dct_type=2):
        mfccs = scipy.fftpack.dct(mel, axis=-2, type=dct_type, norm='ortho')[..., :num_mfcc, :]
        return np.mean(mfccs.T, axis=0)

    @staticmethod
    def get_processed_filelist(path, specaugment=False, specaugment_gamma=1.0):
        lst = []
        melspecs_lst = []
        with open(path, 'r') as fh:
            lines = fh.readlines()
            lines = lines[1:] # skip header
            for line in lines:
                filepath, _, _, _, _, _ = line.split(',')
                filepath = f"features/{filepath}"
                try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
                    # num_mfcc = 40
                    # dct_type = 2
                    # mfccs_librosa = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                    #                                      n_mfcc=num_mfcc).T, axis=0)

                    mel_power = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                    mel_db = librosa.core.power_to_db(mel_power)

                    mfccs = CreateFeatures.mel_to_mfcc(mel_db)
                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    filename = Path(filepath).name
                    filename = int(filename[7:8]) - 1
                    arr = mfccs, filename
                    lst.append(arr)
                    melspecs_lst.append((mel_db, filename))
                    break
                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        # Applies specaugment
        # if specaugment:
        #     print(f"Total dataset size: {len(melspecs_lst)}")
        #     #melspecs_lst = melspecs_lst[: int(specaugment_gamma * len(melspecs_lst))]
        #     print(f"Subsampled size: {len(melspecs_lst)}")
        #     augmented_lst = []
        #     augmentation_count = 0
        #     for mel, label in melspecs_lst:
        #         g = 1 / specaugment_gamma
        #         augmented_spectros = [(CreateFeatures.mel_to_mfcc(mel), label)]
        #         for x in range(math.ceil(g) - 1):
        #             if (math.ceil(g) - g == 0) or (random.random() > math.ceil(g) - g):
        #                 spectro = torch.from_numpy(mel).float()
        #                 spectro = torch.unsqueeze(spectro, 0)
        #                 spectro_aug = time_warp(time_mask(freq_mask(spectro, num_masks=2), num_masks=2))
        #                 spectro_aug = torch.squeeze(spectro_aug)
        #                 spectro_aug = spectro_aug.numpy()
        #                 mfcc = CreateFeatures.mel_to_mfcc(spectro_aug)
        #                 augmented_spectros.append((mfcc, label))
        #                 augmentation_count += 1
        #         augmented_lst += augmented_spectros
        #     lst = augmented_lst
        #     random.shuffle(lst)
        #     print(f"Total Augmented size: {len(lst)} ({augmentation_count} augmented)")
        # return lst

    @staticmethod
    def features_creator(train_filelist, val_filelist, save_dir, specaugment_gamma) -> str:
        """
        This function creates the dataset and saves both data and labels in
        two files, X.joblib and y.joblib in the joblib_features folder.
        With this method, you can persist your features and train quickly
        new machine learning models instead of reloading the features
        every time with this pipeline.
        """

        start_time = time.time()
        lst = CreateFeatures.get_processed_filelist(train_filelist,
                                                    specaugment=specaugment_gamma is not None,
                                                    specaugment_gamma=specaugment_gamma)
        lst_val = CreateFeatures.get_processed_filelist(val_filelist, specaugment=False)


        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
        X, y = zip(*lst)
        X_val, y_val = zip(*lst_val)

        # Array conversion
        X, y = np.asarray(X), np.asarray(y)
        X_val, y_val = np.asarray(X_val), np.asarray(y_val)

        # Array shape check
        print(X.shape, y.shape)
        print(X_val.shape, y_val.shape)

        # Preparing features dump
        X_name, y_name, X_val_name, y_val_name = 'X.joblib', 'y.joblib', 'X_val.joblib', 'y_val.joblib'
        Path(save_dir).mkdir(parents=True)
        Path(f'{save_dir}{X_name}').touch()
        Path(f'{save_dir}{y_name}').touch()
        Path(f'{save_dir}{X_val_name}').touch()
        Path(f'{save_dir}{y_val_name}').touch()
        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))
        joblib.dump(X_val, os.path.join(save_dir, X_val_name))
        joblib.dump(y_val, os.path.join(save_dir, y_val_name))
        return "Completed"


if __name__ == '__main__':
    TRAIN_FILELIST = sys.argv[1]
    VAL_FILELIST = sys.argv[2]
    model_name = sys.argv[3]
    specaugment_gamma = float(sys.argv[4]) if sys.argv[4] != 'none' else None

    print('Routine started')
    FEATURES = CreateFeatures.features_creator(train_filelist=TRAIN_FILELIST,
                                               val_filelist=VAL_FILELIST,
                                               save_dir=SAVE_DIR_PATH+model_name,
                                               specaugment_gamma=specaugment_gamma)
    print('Routine completed.')
