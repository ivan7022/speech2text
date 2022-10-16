from torchaudio import transforms
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, p, frequency, *args, **kwargs):
        self.p = p
        self._aug = transforms.FrequencyMasking(frequency)

    def __call__(self, x: Tensor):
        augment = random.random() < self.p
        return self._aug(x) if augment else x