import sys
sys.path.insert(0, 'src')

from config import load_config
from data import prepare
from train import train_run

cfg = load_config("configs/training.yaml")

# Replace this with your actual folder path where ptbxl_database.csv lives:
cfg['data']['root'] = 'data/ptbxl'  # or e.g., '/media/ehsan/MyFiles2/.../ptbxl'

cfg['run_mode'] = 'pilot'
cache, manifest = prepare(cfg)

run_path = train_run(cfg, cache, architecture='tft', preprocessing='P1', feature_set='F0', resume=False)
print(f"Pilot training completed successfully! Checkpoint saved at: {run_path}")
