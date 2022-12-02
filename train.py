from audio_aug.augment import Augmentor
from model import get_model, train


def main(augmentor: Augmentor):
   model = get_model()
   train(augmentor, model, 150, 16)
