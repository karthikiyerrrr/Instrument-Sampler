# Notebooks

Jupyter notebooks for experimenting with preprocessing, transcription, and synthesis pipelines outside the main application.

## Setup

```bash
pip install jupyterlab
jupyter lab --notebook-dir=notebooks
```

## Organization

Directories mirror the three pipeline stages. Notebooks are prefixed by instrument group (`group1_`, `group2_`, `group3_`) so related work stays grouped together.

| Folder | Purpose |
|---|---|
| `shared/` | Reusable utility modules imported by notebooks |
| `preprocessing/` | Feature extraction, noise reduction, HPSS, spectral gating |
| `transcription/` | basic-pitch, onset detection, MIDI evaluation |
| `synthesis/` | DDSP, Karplus-Strong, WaveNet/NSynth prototyping |

## Shared Utilities

The `shared/` directory contains Python modules that notebooks import via:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

from shared.ddsp import preprocess_audio, finetune, run_inference
from shared.transcription import transcribe_wav
```

| Module | Contents |
|---|---|
| `shared/ddsp.py` | Model download, f0/loudness preprocessing, fine-tuning loop, inference |
| `shared/transcription.py` | basic-pitch ONNX transcription |

## Current Notebooks

| Notebook | Group | Description |
|---|---|---|
| `preprocessing/group1_ddsp_preprocess.ipynb` | 1 | Download DDSP models, extract f0/loudness, chunk for training |
| `transcription/group1_basic_pitch.ipynb` | 1 | Monophonic transcription with basic-pitch, MIDI inspection |
| `synthesis/group1_ddsp_finetune.ipynb` | 1 | Fine-tune DDSP Autoencoder on calibration audio |
| `synthesis/group1_ddsp_inference.ipynb` | 1 | Run DDSP timbre transfer, resample to 44.1 kHz |

## Naming Convention

```
{group}_{engine_or_method}[_stage].ipynb
```

Examples:
- `group1_ddsp_preprocess.ipynb` — Group 1, DDSP engine, preprocessing stage
- `group2_hpss.ipynb` — Group 2, HPSS preprocessing (future)
- `group3_wavenet_train.ipynb` — Group 3, WaveNet training (future)
