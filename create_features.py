"""
This files creates the X and y features in joblib to be used by the predictive models.
"""

import os
import sys
import time
import joblib
import librosa
import numpy as np
from pathlib import Path

from config import SAVE_DIR_PATH
from config import TRAINING_FILES_PATH


class CreateFeatures:
    @staticmethod
    def get_processed_filelist(path):
        lst = []
        with open(path, 'r') as fh:
            lines = fh.readlines()
            lines = lines[1:] # skip header
            for line in lines:
                filepath, _, _, _, _, _ = line.split(',')
                try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    X, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                         n_mfcc=40).T, axis=0)
                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    file = int(file[7:8]) - 1
                    arr = mfccs, file
                    lst.append(arr)
                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue
        return lst

    @staticmethod
    def features_creator(train_filelist, val_filelist, save_dir) -> str:
        """
        This function creates the dataset and saves both data and labels in
        two files, X.joblib and y.joblib in the joblib_features folder.
        With this method, you can persist your features and train quickly
        new machine learning models instead of reloading the features
        every time with this pipeline.
        """

        start_time = time.time()
        lst = CreateFeatures.get_processed_filelist(train_filelist)
        lst_val = CreateFeatures.get_processed_filelist(val_filelist)
        # for subdir, dirs, files in os.walk(path):
        #     for file in files:
        #         try:
        #             # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        #             X, sample_rate = librosa.load(os.path.join(subdir, file),
        #                                           res_type='kaiser_fast')
        #             mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
        #                                                  n_mfcc=40).T, axis=0)
        #             # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        #             # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        #             file = int(file[7:8]) - 1
        #             arr = mfccs, file
        #             lst.append(arr)
        #         # If the file is not valid, skip it
        #         except ValueError as err:
        #             print(err)
        #             continue

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
        Path(save_dir).mkdir()
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
    print('Routine started')
    FEATURES = CreateFeatures.features_creator(train_filelist=TRAIN_FILELIST, val_filelist=VAL_FILELIST, save_dir=SAVE_DIR_PATH)
    print('Routine completed.')
