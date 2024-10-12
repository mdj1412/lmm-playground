# LMM Phi-3-Vision Pipline (Large Multimodal Model)

## Installation

```
cd lmm-playground
conda create --name lmm_pipe python=3.8
conda activate lmm_pipe
pip install -r requirements.txt
```

## Data Preparation
```
python generate_dataset.py --dataset-name RefCOCO --split val
python generate_dataset.py --dataset-name RefCOCOplus --split val
python generate_dataset.py --dataset-name RefCOCOg --split val
```

## Phi-3-Vision Training

### SFT
```
# Example
python phi3vision_train.py --dataset-name RefCOCO --split val --epochs 3 --save-dir saved_models/sft --loss_fn sft
```

### Digit base loss
```
# Example
python phi3vision_train.py --dataset-name RefCOCO --split val --epochs 3 --save-dir saved_models/digit_loss --loss_fn digit_base --learning-rate  0.0001
```

### Digit loss
```
# Example
python phi3vision_train.py --dataset-name RefCOCO --split val --epochs 3 --save-dir saved_models/digit_loss --loss_fn digit --learning-rate  0.0001
```

## TODO
* Current implementation: Trained on RefCOCO, RefCOCOg, RefCOCOplus validation sets -> Change to: Training on train datasets
* Florence-2, Paligemma post-train

## Reference
* [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)
* [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726)
* [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)
* [Ferret-v2: An Improved Baseline for Referring and Grounding with Large Language Models](https://arxiv.org/abs/2404.07973)
* [Phi-3CookBook](https://github.com/microsoft/Phi-3CookBook/tree/main/code/04.Finetuning)