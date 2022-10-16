from typing import List, NamedTuple

import torch
from pyctcdecode import build_ctcdecoder

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_path='4-gram.arpa'):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.ctc_decoder = build_ctcdecoder(
            [''] + list(map(str.upper, self.alphabet)),
            lm_path,
            alpha=0.2,
            beta=1e-3, 
        )

    def ctc_decode(self, inds: List[int]) -> str:
        res = []
        for idx in inds:
            if res and res[-1] == self.ind2char[idx]: continue
            res.append(self.ind2char[idx])
        return ''.join(tok for tok in res if tok != self.EMPTY_TOK).lstrip()

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        probs = probs[:probs_length, :].cpu().detach().numpy()
        decoded = self.ctc_decoder.decode_beams(probs, beam_width=beam_size)

        for words, _, _, _, weighted_prob in decoded:
            word = words.lower()
            hypos.append(Hypothesis(word, weighted_prob))

        return hypos
