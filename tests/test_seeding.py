import multiprocessing

import numpy as np
import pytest

from ptbxl_forecast.seeding import seed_everything, torch_generator

torch = pytest.importorskip("torch")


class Toy(torch.utils.data.Dataset):
    """Synthetic 12-lead windows; noise is drawn inside workers to exercise worker seeding."""

    def __init__(self, n: int = 64):
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((n, 70, 12)).astype(np.float32)
        self.y = rng.standard_normal((n, 30, 12)).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        noise = np.random.normal(0, 0.01, self.x[i].shape).astype(np.float32)
        return torch.from_numpy(self.x[i] + noise), torch.from_numpy(self.y[i])


class TinyCnnLstm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(12, 16, 5, padding=2)
        self.lstm = torch.nn.LSTM(16, 16, batch_first=True)
        self.drop = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(16, 30 * 12)

    def forward(self, x):
        h = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(self.drop(out[:, -1])).view(-1, 30, 12)


def train(seed: int, num_workers: int, context: str | None = None) -> torch.Tensor:
    seed_everything(seed)
    model = TinyCnnLstm()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        Toy(),
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        generator=torch_generator(seed),
        multiprocessing_context=context,
    )
    for _ in range(2):
        for x, y in loader:
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_training_is_bitwise_reproducible():
    a = train(seed=42, num_workers=0)
    assert torch.equal(a, train(seed=42, num_workers=0))
    assert not torch.equal(a, train(seed=43, num_workers=0))


# Linux defaults to fork; Windows and macOS use spawn. Test every method this OS offers.
@pytest.mark.parametrize("context", multiprocessing.get_all_start_methods())
def test_reproducible_with_multiple_workers(context):
    a = train(seed=42, num_workers=2, context=context)
    assert torch.equal(a, train(seed=42, num_workers=2, context=context))


def test_numpy_generator_is_seeded():
    assert seed_everything(7).random() == seed_everything(7).random()
