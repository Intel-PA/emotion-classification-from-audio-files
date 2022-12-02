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

    rd = RavdessDataset(batch_sz, TESS_ORIGINAL_FOLDER_PATH)
    rd.load_process()
    t, v = rd.split(val_pct=0.33)
    optimiser = keras.optimizers.RMSprop()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    datasets = {
        'train': t,
        'val': v
    }
    metrics = {
        'train': keras.metrics.SparseCategoricalAccuracy(),
        'val': keras.metrics.SparseCategoricalAccuracy()
    }
    for epoch in range(epochs):
        for mode in ['train', 'val']:
            print("\nStart of epoch %d" % (epoch,))
            dataset = datasets[mode]
            metric = metrics[mode]
            is_train = (mode == "train")

            for step, (x_batch, y_batch) in enumerate(dataset):
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

                # Log every 200 batches.
                # if step % 200 == 0:
                #     print(f"{mode} loss (for one batch) at step {step}: {float(loss_value):.4f}")
                #     print(f"Seen so far: {(step + 1) * batch_sz} samples")

            # Display metrics at the end of each epoch.
            acc = metric.result()
            print(f"{mode} acc : {float(acc)}, loss: {float(loss_value)}")

            # Reset training metrics at the end of each epoch
            metric.reset_states()
    wandb.join()

if __name__ == '__main__':
    train(150, 16)
