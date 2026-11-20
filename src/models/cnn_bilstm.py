"""CLASSIC family: 1D-CNN feature extractor followed by a bidirectional LSTM."""
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels, n_classes, channels=(64, 128), kernel=5,
                 lstm_hidden=64, dropout=0.3, use_lstm=True):
        super().__init__()
        layers, c = [], in_channels
        for co in channels:
            layers += [nn.Conv1d(c, co, kernel, padding=kernel // 2), nn.BatchNorm1d(co),
                       nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout * 0.5)]
            c = co
        self.cnn = nn.Sequential(*layers)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(c, lstm_hidden, batch_first=True, bidirectional=True)
            feat = 2 * lstm_hidden
        else:
            feat = c
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feat, n_classes)

    def forward(self, x):                      # x: (B, T, C)
        h = self.cnn(x.transpose(1, 2))        # (B, c, T')
        if self.use_lstm:
            o, _ = self.lstm(h.transpose(1, 2))
            f = o.mean(1)
        else:
            f = h.mean(2)
        return self.fc(self.drop(f))
