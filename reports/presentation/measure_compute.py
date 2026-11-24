"""CPU cost per architecture (4 threads, random weights, Phase 9/10 configurations):
parameters, peak RSS during a training step (batch 64), training-step time and
inference time per record (batch 512). Each model runs in a fresh process."""

import json
import resource
import subprocess
import sys
import time

NAMES = ["cnn", "lstm", "cnn_lstm", "patchtst", "transformer", "sn4_gated"]


def one(name: str) -> dict:
    import torch

    from ptbxl_forecast.models import MODELS, GatedBaselineCNNLSTM

    torch.set_num_threads(4)
    torch.manual_seed(0)
    if name == "sn4_gated":
        model, extra = GatedBaselineCNNLSTM(24, 12, 290), 12
    else:
        model, extra = MODELS[name](in_channels=24, out_leads=12, horizon=290), 0
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    x, y = torch.randn(64, 700, 24 + extra), torch.randn(64, 290, 12)
    times = []
    for i in range(6):
        t = time.perf_counter()
        opt.zero_grad()
        torch.nn.functional.l1_loss(model(x), y).backward()
        opt.step()
        if i:
            times.append(time.perf_counter() - t)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model.eval()
    xi = torch.randn(512, 700, 24 + extra)
    with torch.no_grad():
        model(xi)
        t = time.perf_counter()
        for _ in range(3):
            model(xi)
        inf = (time.perf_counter() - t) / 3 / 512
    return {"model": name, "params": sum(p.numel() for p in model.parameters()),
            "train_step_s_batch64": sorted(times)[len(times) // 2],
            "peak_rss_train_MB": peak / 1024, "rss_increase_train_MB": (peak - base) / 1024,
            "inference_ms_per_record": inf * 1000}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(one(sys.argv[1])))
    else:
        out = [json.loads(subprocess.run([sys.executable, __file__, n], capture_output=True, text=True,
                                         check=True).stdout.strip().splitlines()[-1]) for n in NAMES]
        json.dump(out, open("reports/presentation/compute.json", "w"), indent=2)
        for r in out:
            print(r)
