import wandb
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import RMSprop, SGD

from data_util import load_data, mel_fn, mel_to_mfcc, batch_mel_to_mfcc, batch_mel_fn
from data_util import RAVDESSDataset
from config import TESS_ORIGINAL_FOLDER_PATH
from pytorch_lightning.loggers import WandbLogger


def get_model(augmentor, mel_fn):
    return PL_model(1, 8, augmentor, mel_fn)


def zip_data_batch(batch):
    signals, labels = batch
    signals_list = list(signals.cpu().numpy())
    labels_list = list(labels.cpu().numpy())
    return zip(signals_list, labels_list)


class PL_model(pl.LightningModule):
    def __init__(self, in_channels, outputs, augmentor=None, mel_fn=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=64,
                      padding=2,
                      kernel_size=5
                      ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 40, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
        )
        self.augmentor = None if augmentor is None else augmentor.get(mel_fn)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.001, momentum=0.9)

    def training_step(self, batch, batch_idx):
        mfcc_batch, label_batch = batch
        batch_sz = len(mfcc_batch)
        mfcc_batch = mfcc_batch.unsqueeze(1)
        output = self.model(mfcc_batch)
        loss = nn.CrossEntropyLoss()(output, label_batch)

        preds = torch.argmax(output, dim=1)
        correct = int(torch.eq(preds, label_batch).sum())
        minibatch_acc = correct / batch_sz

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc',  minibatch_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mfcc_batch, label_batch = batch
        batch_sz = len(mfcc_batch)
        mfcc_batch = mfcc_batch.unsqueeze(1)
        output = self.model(mfcc_batch)
        loss = nn.CrossEntropyLoss()(output, label_batch)

        preds = torch.argmax(output, dim=1)
        correct = int(torch.eq(preds, label_batch).sum())
        minibatch_acc = correct / batch_sz

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc',  minibatch_acc, on_step=False, on_epoch=True)
        return loss


def train(augmentor, model, epochs, batch_sz):
    run_name_elements = augmentor.config["run_name"].split("_")
    project_name = "_".join(run_name_elements[:2])  # Use experiment and kind names
    wandb.init(project=project_name, reinit=True)
    wandb.run.name = augmentor.config["run_name"]

    train_dl, val_dl = load_data(TESS_ORIGINAL_FOLDER_PATH, augmentor, batch_sz)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         precision=32,
                         max_epochs=epochs,
                         logger=wandb_logger
                         )
    trainer.fit(model, train_dl, val_dl)
    wandb.join()


if __name__ == '__main__':
    train(150, 16)
# rd = RavdessDataset(batch_sz, TESS_ORIGINAL_FOLDER_PATH, augmenter=augmentor)
# optimiser = keras.optimizers.RMSprop()
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# datasets = {
#     'train': {
#         'dataset': rd.train_ds,
#         'get_batch_fn': rd.get_train_batch
#     },
#     'val': {
#         'dataset': rd.val_ds,
#         'get_batch_fn': rd.get_val_batch
#     },
# }
# metrics = {
#     'train': keras.metrics.SparseCategoricalAccuracy(),
#     'val': keras.metrics.SparseCategoricalAccuracy()
# }
# for epoch in range(epochs):
#     for mode in ['train', 'val']:
#         print("\nStart of epoch %d" % (epoch,))
#         dataset = datasets[mode]['dataset']
#         get_batch_fn = datasets[mode]['get_batch_fn']
#         metric = metrics[mode]
#         is_train = (mode == "train")
#
#         for step, _ in enumerate(dataset):
#             (x_batch, y_batch) = get_batch_fn()
#             if is_train:
#                 with tf.GradientTape() as tape:
#                     logits = model(x_batch, training=is_train)
#                     loss_value = loss_fn(y_batch, logits)
#
#                     grads = tape.gradient(loss_value, model.trainable_weights)
#                     optimiser.apply_gradients(zip(grads, model.trainable_weights))
#             else:
#                 logits = model(x_batch, training=is_train)
#                 loss_value = loss_fn(y_batch, logits)
#
#             # Update metric.
#             metric.update_state(y_batch, logits)
#
#         # Display metrics at the end of each epoch.
#         acc = metric.result()
#         wandb.log({f"{mode}_loss": loss_value, f"{mode}_acc": acc}, step=epoch)
#         print(f"{mode} acc : {float(acc)}, loss: {float(loss_value)}")
#
#         # Reset training metrics at the end of each epoch
#         metric.reset_states()
# def get_model():

# inputs = keras.Input(shape=(40, 1))
# outputs = Conv1D(64, 5, padding='same',
#                  input_shape=(40, 1))(inputs)
# outputs = Activation('relu')(outputs)
# outputs = Dropout(0.2)(outputs)
# outputs = Flatten()(outputs)
# outputs = Dense(8)(outputs)
# outputs = Activation('softmax')(outputs)
# model = keras.Model(inputs=inputs, outputs=outputs)
# return model
