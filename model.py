import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
import numpy as np



inputs = Conv1D(64, 5, padding='same',
                 input_shape=(40, 1)) 

outputs = Activation('relu')(inputs)
outputs = Dropout(0.2)(outputs)
outputs = Flatten()(outputs)
outputs = Dense(8)(outputs)
outputs = Activation('softmax')(outputs)
model = keras.Model(inputs=inputs, outputs=outputs)

optim = keras.optimizers.RMSprop()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
datasets = {
    'train': 1,
    'val': 2
}
metrics = {
    'train': keras.metrics.SparseCategoricalAccuracy(),
    'val': keras.metrics.SparseCategoricalAccuracy()
}

def train(epochs, batch_sz, datasets, metrics, optimiser, loss_fn):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_sz))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_ds:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))