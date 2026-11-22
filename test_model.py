import sys
sys.path.insert(0, 'src')

import torch
from models import build_model

for arch in ['tft', 'patchtst']:
    model = build_model(architecture=arch, static_dim=4, horizon=300)
    model.eval()
    pred = model(torch.zeros(2, 700, 12), torch.zeros(2, 4))
    print(f"Architecture: {arch.upper()} | Output Shape: {tuple(pred.shape)}")
    assert tuple(pred.shape) == (2, 300, 12), f"Shape mismatch for {arch}!"
