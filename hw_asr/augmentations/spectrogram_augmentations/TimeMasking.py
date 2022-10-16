from torchaudio import transforms
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p, time, *args, **kwargs):
        self.p = p
        self._aug = transforms.TimeMasking(time)

    def __call__(self, x: Tensor):
        augment = random.random() < self.p
        return self._aug(x) if augment else x