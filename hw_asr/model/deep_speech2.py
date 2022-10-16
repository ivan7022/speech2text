from torch import nn

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.fc_hidden = fc_hidden
        # BxFxT
        self.preproj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        self.gru = nn.GRU(n_feats*32//4, fc_hidden, num_layers=4, batch_first=True, bidirectional=True)

        self.postconv = nn.Sequential(
            nn.Conv1d(fc_hidden * 2, fc_hidden * 2, 21, groups=fc_hidden * 2, padding=10, bias=False),
            nn.BatchNorm1d(fc_hidden * 2),
        )

        self.postpoj = nn.Linear(fc_hidden * 2, n_class)


    def forward(self, spectrogram, *args, **kwargs):
        x = self.preproj(spectrogram.unsqueeze(1))
        x = x.permute((0, 3, 1, 2)) # BxCxFxT -> BxTxCxF
        x, _ = self.gru(x.reshape(x.shape[0], x.shape[1], -1)) # BxTx(CxF)
        x = self.postconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.postpoj(x)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - 1) // 2 + 1