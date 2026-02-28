# Capstone Project: Activation Functions Benchmark

This project benchmarks 9 activation functions across 2 deep learning architectures:
- **ResNet-34** on **CIFAR-10** (Computer Vision)
- **Transformer Encoder** on **AG News** (NLP)

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `activations.py`: Custom activation functions (Mish, Swish, etc.) and registry.
- `models/`: Contains model definitions (`resnet.py`, `transformer.py`).
- `data.py`: Data loading and preprocessing scripts.
- `train_utils.py`: Training loop, metrics calculation (FLOPs, Memory, Accuracy).
- `main.py`: Main script to run all 18 experiments.

## Running Experiments

To run the full suite of experiments:
```bash
python main.py
```

This will:
1. Download datasets (CIFAR-10, AG News).
2. Train models using a **fixed Train/Validation/Test split (90% Train, 10% Val of the official Training set, plus official Test set)**.
3. Train for **30 epochs** per experiment (reduced from 50 for efficiency).
4. Save checkpoints to `checkpoints/` and detailed results to `results/`.
5. Use a fixed random seed (`42`) for full reproducibility.

## Metrics
- Accuracy
- Validation Loss
- Training Time
- FLOPs (Floating Point Operations)
- Gradient Norm
- Peak GPU Memory Usage

## Activation Functions Compared
- ReLU
- LeakyReLU
- ELU
- SELU
- GELU
- Swish (SiLU)
- Mish
- Hardswish
- Softplus

## Notes
- Ensure CUDA is available for faster training.
- Adjust `batch_size` in `main.py` if encountering OOM errors.
- Random seeds are fixed to `42` across `main.py` and `train_utils.py`.
