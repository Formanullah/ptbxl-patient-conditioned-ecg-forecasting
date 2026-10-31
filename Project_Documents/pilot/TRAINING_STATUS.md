# Training progress

Profile: pilot. Validation MSE is standardized; test MAE is in original mV.
Only completed models for this cohort have test metrics.

| Run | State | Epochs | Best validation MSE | Test MAE (mV) |
| --- | --- | --- | --- | --- |
| lstm_ecg_only | complete | 5 | 0.652309 | 0.151706 |
| lstm_demographics | complete | 5 | 0.662539 | 0.151759 |
| lstm_patient_history | complete | 5 | 0.663526 | 0.150113 |
| lstm_patient_history_contrastive | complete | 5 | 0.662394 | 0.151782 |
| cnn_lstm_ecg_only | complete | 5 | 0.679288 | 0.154296 |
| cnn_lstm_demographics | complete | 5 | 0.679670 | 0.155650 |
| cnn_lstm_patient_history | complete | 5 | 0.679014 | 0.154251 |
| cnn_lstm_patient_history_contrastive | complete | 5 | 0.689418 | 0.155719 |

Resume remaining work with:

```powershell
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/pilot.yaml
```

Rerunning this command resumes last completed epochs and skips measured completed runs.
Regenerate with `python tools/status.py --config configs/pilot.yaml`.
