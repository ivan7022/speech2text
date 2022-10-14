import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    for dataset_item in dataset_items:
        for key, value in dataset_item.items():
            result_batch[key] = result_batch.get(key, []) + [
                value.transpose(0, -1).squeeze(-1) if torch.is_tensor(value) else value
            ]
            if key in ['text_encoded', 'spectrogram']:
                result_batch[f'{key}_length'] = result_batch.get(
                    f'{key}_length', []
                ) + [torch.tensor([value.shape[-1]])]
    for key in result_batch:
        if result_batch[key] and torch.is_tensor(result_batch[key][0]):
            result_batch[key] = torch.nn.utils.rnn.pad_sequence(
                [seq for seq in result_batch[key]], True
            ).transpose(1, -1).squeeze()
            
    return result_batch