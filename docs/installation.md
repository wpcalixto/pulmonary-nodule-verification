# Installation

1. Python 3.11 or later, Linux, NVIDIA GPU with CUDA 12.8 for stages 08–10 and 14/22b (GPU filter maps).
2. `python -m venv .venv && . .venv/bin/activate`
3. `pip install -r requirements.txt` (PyTorch: install the CUDA 12.8 wheel matching your driver first).
4. nnU-Net v2.8.1: copy the files of `trainers/` into `nnunetv2/training/nnUNetTrainer/` and apply `patches/polylr.py`
   as described in `patches/patches_manifest.json`.
5. Set `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` (nnU-Net convention) and `PULMONARY_BASE`.
6. Run `python tests/test_smoke.py`.

Recorded environments: `environment/training_lock.json` (training), `environment/pip_freeze_inference_and_analysis.txt` (analysis).
