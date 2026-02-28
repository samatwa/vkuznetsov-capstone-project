import math
import torch
import torch.nn as nn
from activations import get_activation


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken,
        d_model,
        nhead,
        d_hid,
        nlayers,
        num_classes,
        activation="relu",
        dropout=0.5,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"

        act_module = get_activation(activation)

        # batch_first=True: форма вхідних/вихідних даних [batch, seq_len, d_model]
        # Це вмикає Fast Path у PyTorch (ядра FlashAttention на GPU) та усуває попередження scaled_dot_product_attention.
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            activation=act_module,
            batch_first=True,
            norm_first=True,  # Pre-LN: стабільніше навчання для функцій активації, відмінних від ReLU
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=0)
        # Позиційне кодування, що навчається — простіше та часто відповідає якості синусоїдального
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        nn.init.trunc_normal_(self.pos_encoder, std=0.02)

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask=None):
        # src: [batch, seq_len]  (batch_first=True у data.py collate_fn)
        seq_len = src.size(1)

        x = self.encoder(src) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)

        # Створення маски заповнення з токенів <pad>=0, щоб увага ігнорувала їх
        if src_key_padding_mask is None:
            src_key_padding_mask = src == 0  # True where padding

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Mean-pool тільки по позиціях без заповнення
        mask = ~src_key_padding_mask  # [batch, seq_len]
        lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        output = (output * mask.unsqueeze(-1)).sum(dim=1) / lengths  # [batch, d_model]

        return self.decoder(output)
