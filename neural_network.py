"""
Neural network train file.
"""
import os
import joblib
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

from config import SAVE_DIR_PATH
from config import MODEL_DIR_PATH
from config import MODEL_NAME

class TrainModel:

    @staticmethod
    def train_neural_network(X_train, y_train, X_test, y_test, run_name) -> None:
        """
        This function trains the neural network.
        """
        wandb.init(project="emoclass_soxaugment_performance", reinit=True)
        wandb.run.name = run_name
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        print(x_traincnn.shape, x_testcnn.shape)

        model = Sequential()
        model.add(Conv1D(64, 5, padding='same',
                         input_shape=(40, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Activation('softmax'))

        print(model.summary)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        cnn_history = model.fit(x_traincnn, y_train,
                                batch_size=16, epochs=5,
                                validation_data=(x_testcnn, y_test),
                                callbacks=[WandbCallback()])

        # Loss plotting
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(cnn_history.history['acc'])
        plt.plot(cnn_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')

        predict_x = model.predict(x_testcnn)
        predictions = np.argmax(predict_x, axis=1)
        new_y_test = y_test.astype(int)
        matrix = confusion_matrix(new_y_test, predictions)

        print(classification_report(new_y_test, predictions))
        print(matrix)

        model_name = 'Emotion_Voice_Detection_Model.h5'

        # Save model and weights
        if not os.path.isdir(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    print('Training started')
    X = joblib.load(SAVE_DIR_PATH + 'X.joblib')
    y = joblib.load(SAVE_DIR_PATH + 'y.joblib')
    TrainModel.train_neural_network(X=X, y=y, run_name=MODEL_NAME)
