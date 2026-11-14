# Historical implementation

This snapshot preserves the source, configurations, tests, notebooks and documents
at commit `2ad3e680fbe408345212f5d3b18bda2a115a1b45`. Existing historical caches,
weights and outputs remain at their original project-root locations. Their hashes
are recorded in `protected_artifacts.json`.

Run historical scripts from the **main project root**, giving the archived config
explicitly, for example:

```sh
python archive/legacy_2ad3e68/src/evaluate.py --config archive/legacy_2ad3e68/configs/pilot.yaml --model lstm --variant patient_history
```

Use an environment compatible with the recorded original dependencies. This is
a historical protocol, not a new-run result or an instruction to evaluate fold 10
while tuning the new study. Do not change old checkpoint/configuration files.

The 500-Hz pilot used 300 observed / 150 future samples and five epochs. Historical
test MAEs were persistence 0.141546, history LSTM 0.150113, and history CNN-LSTM
0.154251 mV. These do not measure the new 100-Hz, 700/300 task. Other archived
context-length and preprocessing studies also retain their own protocols.

Some old active documentation references expanded-cohort artifacts absent from
this checkout. Those claims cannot be independently reproduced from the available
files and are not inserted into new comparisons.

The earlier exploratory notebook, slide decks, and slide builder are also in
this archive. The exploratory notebook's generated figures and tables are still
under `ptbxl_eda_outputs/` at the main project root. To rerun it from the archive,
set `BASE_DIR` to your dataset and choose a separate `OUTPUT_DIR` so the original
figures and tables stay intact.
