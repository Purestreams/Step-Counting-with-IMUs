# Required Class and Strict Output Format

Your file must define the class below with the same method names and return keys. The grader will call `run_offline`.

```python
import numpy as np

class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    You can add any other attributes you need to the class. But you should not change the
    interface of the class.
    """
    
    def __init__(self):
        """Initialize the step counter."""
        raise NotImplementedError
    
    def reset(self) -> None:
        """
        Reset internal state such as buffers and cumulative count.
        After reset(), total_steps should be 0.
        """
        raise NotImplementedError
    
    def update(self, data_chunk: dict) -> dict:
        """
        Real-time update: process a chunk of new samples.
        
        **Input**
        - `data_chunk["time"]`: numpy.ndarray with shape (M,) [required]
        - `data_chunk["acc"]`: numpy.ndarray with shape (M, 3) in m/s² [required]
        - `data_chunk["gyro"]`: numpy.ndarray with shape (M, 3) in rad/s [optional]
        - `data_chunk["mag"]`: numpy.ndarray with shape (M, 3) in uT [optional]
        
        Chunks arrive sequentially.
        
        **Output** (must contain all keys)
        ```python
        {
            "new_steps": int,
            "total_steps": int,
            "new_step_timestamps": np.ndarray,  # shape (K,), float seconds
            "diagnostics": dict
        }
        ```
        """
        raise NotImplementedError
    
    def run_offline(self, data: dict) -> dict:
        """
        Offline processing: process a full recording.
        
        **Input**
        - `data["time"]`: numpy.ndarray with shape (N,) [required]
        - `data["acc"]`: numpy.ndarray with shape (N, 3) in m/s² [required]
        - `data["gyro"]`: numpy.ndarray with shape (N, 3) in rad/s [optional]
        - `data["mag"]`: numpy.ndarray with shape (N, 3) in uT [optional]
        
        **Output** (must contain all keys)
        ```python
        {
            "step_count": int,
            "step_timestamps": np.ndarray,  # shape (K,), float seconds
            "diagnostics": dict
        }
        ```
        
        **Requirements on output:**
        - `"step_count"` must be a Python int and must be ≥ 0.
        - `"step_timestamps"` must be a 1D NumPy array of dtype float with shape (K,).
          Each entry is a timestamp in seconds. If your algorithm does not produce
          timestamps, return an empty array with shape (0,) rather than None.
        - `"diagnostics"` must be a Python dict. It may be empty.
        """
        raise NotImplementedError
```

## Strict Requirements

- Do not change the class name or method names
- Do not print inside any method
- Do not read files inside any method
- Do not modify the input dictionaries
- Do not assume optional keys (gyro, mag) always exist

## Algorithm Requirements

### Required Functionality

- Use 3-axis accelerometer data as the required main sensing input
- May use gyroscope/magnetometer as optional auxiliary signals
- Work automatically on different recordings without per-file manual tuning
- Be robust to phone orientation changes
- Run correctly when only time + acc are provided
- Output a non-negative integer step count in offline mode
- Maintain correct cumulative counting in real-time mode

### Restrictions

- Do not use pretrained machine learning models
- Do not use built-in step counter libraries or phone step APIs
- Do not hard-code recording-specific parameters
- Standard numerical libraries (NumPy, SciPy, matplotlib) are allowed

### Implementation Hints

- Start with peak-detection baseline on |a_mag(t)|
- Use |a_mag(t)| for orientation robustness
- Apply smoothing (moving average or low-pass filtering)
- In streaming mode, keep a short buffer for stability across chunk boundaries
- Keep method-specific state (buffers, timers, thresholds) inside the StepCounter object


## Deliverables

### Real-Time Demonstration
Create a demo using phyphox (Remote Access) showing:
1. Real-time data curve that updates continuously
2. Real-time step counts (immediate detection + cumulative count)

### Report
Submit a PDF report including:
- Data collection method and ground-truth step count recording
- Algorithm description and key parameters
- Results on own recordings with error metrics
- Screenshots of real-time visualization
- Comparison of multiple methods (if explored)
- Discussion of limitations and failure cases