import numpy as np
import torch
import torch.nn as nn

from .taxonomy import COARSE, FINE_HEADS, LANGS


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TinyFC(nn.Module):
    """Small CNN, ~0.75M parameters. Input (B, 1, 64, 256) ink in [0, 1].

    One shared backbone with three kinds of heads:
      coarse  4 classes common to Chinese, Japanese and Korean
      script  ja (kana-mixed) / zh (Han only) / ko (Hangul): routes to a fine head
      fine    one small head per enabled (language, coarse) pair
    """

    def __init__(self, fine_heads=FINE_HEADS, widths=(32, 64, 128, 160)):
        super().__init__()
        layers, cin = [], 1
        for i, w in enumerate(widths):
            layers.append(block(cin, w))
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        d = 2 * cin
        self.coarse = nn.Linear(d, len(COARSE))
        self.script = nn.Linear(d, len(LANGS))
        self.fine = nn.ModuleList(nn.Linear(d, len(c)) for c in fine_heads.values())
        # Japanese kana/kanji size ratio of the design (standardized), see taxonomy.TRAIT_CUTS
        self.trait = nn.Linear(d, 1)

    def forward(self, x):
        f = self.features(x)
        f = self.drop(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return self.coarse(f), self.script(f), [h(f) for h in self.fine], self.trait(f).squeeze(1)


TRAIT_MEAN, TRAIT_STD = 0.72, 0.08  # standardization of the kana ratio target


def to_tensor(batch_u8: np.ndarray) -> torch.Tensor:
    """uint8 dark-on-light images (B, H, W) -> float ink tensor (B, 1, H, W)."""
    return (1.0 - torch.from_numpy(np.ascontiguousarray(batch_u8).copy()).float().div_(255.0)).unsqueeze(1)
