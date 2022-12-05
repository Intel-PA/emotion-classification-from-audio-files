import wandb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation

from data_util import RavdessDataset
from config import TESS_ORIGINAL_FOLDER_PATH


def get_model():
    inputs = keras.Input(shape=(40, 1))
    outputs = Conv1D(64, 5, padding='same',
                     input_shape=(40, 1))(inputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(8)(outputs)
    outputs = Activation('softmax')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train(augmentor, model, epochs, batch_sz):
    run_name_elements = augmentor.config["run_name"].split("_")
    project_name = "_".join(run_name_elements[:2])  # Use experiment and kind names
    wandb.init(project=project_name, reinit=True)
    wandb.run.name = augmentor.config["run_name"]

    rd = RavdessDataset(batch_sz, TESS_ORIGINAL_FOLDER_PATH, augmenter=augmentor)
    optimiser = keras.optimizers.RMSprop()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    datasets = {
        'train': {
            'dataset': rd.train_ds,
            'get_batch_fn': rd.get_train_batch
        },
        'val': {
            'dataset': rd.val_ds,
            'get_batch_fn': rd.get_val_batch
        },
    }
    metrics = {
        'train': keras.metrics.SparseCategoricalAccuracy(),
        'val': keras.metrics.SparseCategoricalAccuracy()
    }
    for epoch in range(epochs):
        for mode in ['train', 'val']:
            print("\nStart of epoch %d" % (epoch,))
            dataset = datasets[mode]['dataset']
            get_batch_fn = datasets[mode]['get_batch_fn']
            metric = metrics[mode]
            is_train = (mode == "train")

            for step, _ in enumerate(dataset):
                (x_batch, y_batch) = get_batch_fn()
                if is_train:
                    with tf.GradientTape() as tape:
                        logits = model(x_batch, training=is_train)
                        loss_value = loss_fn(y_batch, logits)

                        grads = tape.gradient(loss_value, model.trainable_weights)
                        optimiser.apply_gradients(zip(grads, model.trainable_weights))
                else:
                    logits = model(x_batch, training=is_train)
                    loss_value = loss_fn(y_batch, logits)

                # Update metric.
                metric.update_state(y_batch, logits)

            # Display metrics at the end of each epoch.
            acc = metric.result()
            wandb.log({f"{mode}_loss": loss_value, f"{mode}_acc": acc}, step=epoch)
            print(f"{mode} acc : {float(acc)}, loss: {float(loss_value)}")

            # Reset training metrics at the end of each epoch
            metric.reset_states()
    wandb.join()

if __name__ == '__main__':
    train(150, 16)
