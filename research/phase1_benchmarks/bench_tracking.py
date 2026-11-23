"""Phase 1 evidence: experiment-tracker overhead. Usage: python bench_tracking.py <mlflow_sqlite|mlflow_file|wandb_offline|tensorboard|jsonl>"""
import os, sys, time, tempfile, json
tool = sys.argv[1]; root = tempfile.mkdtemp(prefix=f"trk_{tool}_"); os.chdir(root)
N_EP, N_MET = 200, 10   # 200 epochs x 10 metrics = 2,000 points per run
t0 = time.perf_counter()
res = {"tool": tool}
try:
    if tool.startswith("mlflow"):
        import mlflow; res["import_s"] = time.perf_counter() - t0
        uri = "sqlite:///" + os.path.join(root, "mlflow.db") if tool == "mlflow_sqlite" else "file://" + os.path.join(root, "mlruns")
        mlflow.set_tracking_uri(uri); mlflow.set_experiment("bench")
        t1 = time.perf_counter()
        for seed in range(3):
            with mlflow.start_run():
                mlflow.log_params({f"p{i}": i for i in range(30)} | {"seed": seed})
                for ep in range(N_EP):
                    mlflow.log_metrics({f"m{j}": ep * j * 1e-3 for j in range(N_MET)}, step=ep)
        res["log_3runs_s"] = time.perf_counter() - t1
        t2 = time.perf_counter(); df = mlflow.search_runs(experiment_names=["bench"])
        res["query_s"] = time.perf_counter() - t2; res["query_rows"] = len(df)
        hist = mlflow.MlflowClient().get_metric_history(df.run_id[0], "m3"); res["history_points"] = len(hist)
    elif tool == "wandb_offline":
        os.environ["WANDB_MODE"] = "offline"; os.environ["WANDB_SILENT"] = "true"
        import wandb; res["import_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        for seed in range(3):
            run = wandb.init(project="bench", dir=root, config={f"p{i}": i for i in range(30)} | {"seed": seed})
            for ep in range(N_EP):
                run.log({f"m{j}": ep * j * 1e-3 for j in range(N_MET)}, step=ep)
            run.finish()
        res["log_3runs_s"] = time.perf_counter() - t1
        try:
            wandb.Api().runs("bench"); res["query_s"] = "ok"
        except Exception as e:
            res["query_s"] = f"FAILS offline: {type(e).__name__}"
    elif tool == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter; res["import_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        for seed in range(3):
            w = SummaryWriter(os.path.join(root, f"run{seed}"))
            for ep in range(N_EP):
                for j in range(N_MET): w.add_scalar(f"m{j}", ep * j * 1e-3, ep)
            w.add_hparams({f"p{i}": i for i in range(30)} | {"seed": seed}, {"final": 0.0}); w.close()
        res["log_3runs_s"] = time.perf_counter() - t1
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        t2 = time.perf_counter(); ea = EventAccumulator(os.path.join(root, "run0")); ea.Reload()
        res["query_s"] = time.perf_counter() - t2; res["history_points"] = len(ea.Scalars("m3"))
    elif tool == "jsonl":
        res["import_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        for seed in range(3):
            d = os.path.join(root, f"run{seed}"); os.makedirs(d)
            json.dump({f"p{i}": i for i in range(30)} | {"seed": seed}, open(f"{d}/params.json", "w"))
            with open(f"{d}/metrics.jsonl", "w") as f:
                for ep in range(N_EP):
                    f.write(json.dumps({"step": ep} | {f"m{j}": ep * j * 1e-3 for j in range(N_MET)}) + "\n"); f.flush()
        res["log_3runs_s"] = time.perf_counter() - t1
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:300]
res["disk_kb"] = int(os.popen(f"du -sk {root}").read().split()[0])
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
