
# Fine-Tune with LoRA

Lab 06 for Advanced Deep Learning (AIGC-5500) at Humber Polytechnic.

This project investigates parameter-efficient fine-tuning using the pretrained HuggingFaceTB/SmolLM2-135M language model for emotion classification.

The implementation compares full fine-tuning against LoRA at different ranks, evaluating classification accuracy, trainable parameters, and memory consumption.

## Setup

**Requirements:** Python 3.12, PyTorch, Transformers, PEFT, Datasets, and Accelerate.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Structure

- `scripts/lab06_starter.py`: Original starter provided by the professor.
- `src/config.py`: Experiment configuration.
- `src/utils.py`: Reproducibility and hardware utilities.
- `src/losses.py`: Custom answer-only loss implementation.
- `src/lora.py`: Custom LoRA implementation.
- `src/verification.py`: LoRA verification utilities.
- `src/experiments.py`: Controlled fine-tuning experiments (planned).
- `src/memory.py`: Memory predictions and measurements (planned).
- `output/figures/`: Generated visualizations.
- `daniel_alejandro_castillo_martin_LAB6.ipynb`: Experimental notebook and analysis.

## Hardware

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Python: 3.12.10
- PyTorch: 2.8.0+cu128
- CUDA: 12.8

## Prediction (Written Before the Rank Sweep)

**Hypothesis 1: LoRA rank 16 vs. full fine-tuning**

I expect LoRA rank 16 to approach the accuracy of full fine-tuning, potentially matching it within the variability observed across random seeds.

A higher adaptation rank provides greater flexibility to represent task-specific weight updates while retaining a relatively small trainable parameter count.

**Hypothesis 2: LoRA rank 1 vs. full fine-tuning**

I expect LoRA rank 1 to perform worse than full fine-tuning and rank 16 because its lower-rank updates have more limited representational capacity.

However, the difference may not be substantial if the emotion classification task requires only a relatively simple adaptation of the pretrained model.

These predictions will be retained regardless of the experimental results.

## Results

Experimental results will be added after the controlled rank sweep.

## Memory Analysis

Predicted and measured memory consumption will be reported for each configuration.

## Conclusion

The final conclusions will be based on the observed accuracy, parameter counts, and memory measurements.

## Known Limitations

The experiments use one relatively small pretrained model, one classification dataset, a fixed training budget, and three random seeds per configuration.
```
