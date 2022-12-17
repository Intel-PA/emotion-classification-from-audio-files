from audio_aug.augment import Augmentor
from model import get_model, train


def main(augmentor: Augmentor, mel_fn):
    model = get_model(augmentor, mel_fn)
    train(augmentor, model, 100, 16)
