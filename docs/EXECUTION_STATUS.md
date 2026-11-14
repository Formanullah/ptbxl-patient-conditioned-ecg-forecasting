# Execution status

| Work | Status |
| --- | --- |
| Real-data software pilot | Completed on CPU with PyTorch 2.15.0+cpu: six two-epoch runs, 339 training ECGs from 300 patients and 81 validation ECGs from 75 patients. Fold 10 waveforms were not opened. |
| Acceptance checks | 26 passed in a separate Python environment. A fresh Jupyter kernel executed all 11 code cells using a labelled synthetic fixture. |
| Workflow checks | Synthetic software checks exercised screening, tuning, grouped permutation, mRMR, four-condition locking, and final evaluation with both baselines. Synthetic scores do not count as PTB-XL results. |
| Data access | The official-download path was verified with one checksum-matched 100 Hz recording and WFDB decoding. The real pilot used the local verified v1.0.3 dataset. |
| Medium screening, tuning, and feature-selection extension | Implemented; real-data research runs pending. |
| Four matched 120-epoch runs and final fold-10 scoring | Implemented; real CUDA execution and final metrics pending. |
| Historical evidence | 1,292 recorded artifact hashes match. The 500 Hz pilot is historical; some reports' extended-cohort supporting files are unavailable. |
| ARIMA | Deferred. |

The real pilot metrics, figures, manifests and checkpoints are under
`results/primary_7s3s_v3/pilot` and in the evidence ZIP. The pilot is too small
and short to rank the research conditions. No final checkpoint has been frozen
on the real dataset.
