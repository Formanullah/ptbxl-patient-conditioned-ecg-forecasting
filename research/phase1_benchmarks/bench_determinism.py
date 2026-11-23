"""Phase 1 evidence: determinism overhead and thread sensitivity.

    python bench_determinism.py speed        # ms/step, deterministic off vs on
    python bench_determinism.py hash <N>     # weight hash after training with N threads
"""
import hashlib, sys, time, torch
from ptbxl_forecast.seeding import seed_everything
mode = sys.argv[1]
class M(torch.nn.Module):  # realistic size: 700x12 -> 300x12, hidden 128
    def __init__(s):
        super().__init__(); s.c = torch.nn.Conv1d(12, 64, 7, padding=3); s.l = torch.nn.LSTM(64, 128, 2, batch_first=True); s.h = torch.nn.Linear(128, 3600)
    def forward(s, x):
        h = torch.relu(s.c(x.transpose(1, 2))).transpose(1, 2)[:, ::2]; o, _ = s.l(h); return s.h(o[:, -1]).view(-1, 300, 12)
def run(det, threads, steps=15):
    torch.set_num_threads(threads); seed_everything(42, deterministic=det)
    m = M(); opt = torch.optim.AdamW(m.parameters(), 1e-3); g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 700, 12, generator=g); y = torch.randn(64, 300, 12, generator=g)
    t = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.mse_loss(m(x), y).backward(); opt.step()
    dt = (time.perf_counter() - t) / steps
    w = torch.cat([p.detach().flatten() for p in m.parameters()])
    return dt, hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16]
if mode == "hash":
    print(run(True, int(sys.argv[2]))[1])
else:
    for det in (False, True):
        r = [run(det, 8)[0] for _ in range(3)]
        print(f"deterministic={det}: {1000*min(r):.1f} ms/step (min of 3, 8 threads)")
