# Silero with Rust

This is a Rust library for Silero VAD, the goal is provide better performance than original version in Python.

## Usage

```python
from silero_rs import SileroVAD

vad = SileroVAD(workers=16, threshold=0.5)
results = vad([audio1, audio2, ...])
```