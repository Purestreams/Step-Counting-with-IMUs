# Submission Package

This folder contains the step-counting implementation, model checkpoint, report PDF, and a reproducible test script.

## 1) Environment initialization

From the project root:

```bash
cd /Users/mio/Vsc/Step-Counting-with-IMUs
python3 -m venv .venv
source .venv/bin/activate
pip install -r submission/requirements.txt
```

## 2) Run the test cases (test1 and test2)

```bash
cd /Users/mio/Vsc/Step-Counting-with-IMUs/submission
python test_step_counter.py
```

This runs on:
- `testdata/test1-84step/Raw Data.csv` (GT = 84)
- `testdata/test2-100steps/Raw Data.csv` (GT = 100)

Results are printed and saved to:
- `submission/test_results.json`

## 3) Use `StepCounter` in your own code

```python
import numpy as np
from step_counter import StepCounter

counter = StepCounter(model_path="artifacts_all_retrain/stepnet_tcn_best.pt", device="auto")

# data format: time -> (N,), acc -> (N, 3)
data = {
    "time": np.array([0.00, 0.02, 0.04], dtype=float),
    "acc": np.array([[0.1, 0.0, -0.1], [0.2, 0.1, -0.1], [0.0, 0.1, -0.2]], dtype=float),
}

result = counter.run_offline(data)
print(result["step_count"])
```

## 4) Included files

- `step_counter.py`
- `infer_nn_step_counter.py`
- `models/tcn_stepnet.py`
- `artifacts_all_retrain/stepnet_tcn_best.pt`
- `report/main.pdf`
- `run_tests.py`
- `test_step_counter.py`
- `test_results.json`
