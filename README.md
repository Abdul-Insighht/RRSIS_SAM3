# RRSIS_SAM3: Referring Remote Sensing Image Segmentation with SAM3

**RRSIS_SAM3** adapts [SAM3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3) for **Referring Remote Sensing Image Segmentation** — given a remote sensing image and a natural language description, the model segments the referred object.

## Key Advantages over RS2-SAM2

| Feature | RS2-SAM2 | RRSIS_SAM3 |
|---------|----------|------------|
| Text Processing | External BEiT-3 encoder | **SAM3 native** text encoder |
| Extra Downloads | SAM2 + BEiT-3 + tokenizer (3 files) | **SAM3 only** (1 file) |
| Text-Vision Fusion | Manual VisionLanguageFusionModule | **Built-in** cross-modal attention |
| Domain Adaptation | Heavy adapter modules | **Lightweight LoRA** (~10M params) |
| Trainable Params | ~200M+ | **~40-50M** |

## Architecture

```
Image (504×504) + Text → SAM3 VL Backbone (ViT + Text Encoder, frozen + LoRA)
                       → Transformer Encoder (text-image fusion, fine-tuned)
                       → DETR Decoder (object detection, fine-tuned)
                       → Segmentation Head (mask prediction, fine-tuned)
                       → Best Mask Selection → Final Segmentation
```

## Supported Datasets

| Dataset | Train | Val | Test | Image Size | Categories |
|---------|-------|-----|------|------------|------------|
| **RRSIS-D** | 12,181 | 1,740 | 3,481 | 800×800 | 20 |
| **RRSIS-HR** | 2,118 | 268 | 264 | 1024×1024 | 7 |
| **RefSegRS** | 2,172 | 413 | 1,817 | 512×512 | — |

All images are resized to **504×504** (divisible by SAM3's ViT patch_size=14).

## Installation

### Option 1: Conda (Recommended)
```bash
git clone https://github.com/your-repo/RRSIS_SAM3.git
cd RRSIS_SAM3
conda env create -f environment.yml
conda activate rrsis_sam3
```

### Option 2: Pip
```bash
git clone https://github.com/your-repo/RRSIS_SAM3.git
cd RRSIS_SAM3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Setup
```python
# In Kaggle notebook:
!pip install timm>=1.0.17 ftfy==6.1.1 regex iopath typing_extensions huggingface_hub einops
!pip install hydra-core tensorboard scipy torchmetrics fvcore fairscale
!pip install scikit-image scikit-learn wandb
```

## Pretrained Weights

SAM3 weights should be placed at `pre-trained-weights/sam3.pt`.

## Data Preparation

```
data/
├── RRSIS-D/                    # RRSIS-D dataset
│   ├── images/
│   ├── masks/
│   └── annotations/
├── RRSIS-HR/                   # RRSIS-HR dataset
│   ├── images/
│   ├── masks/
│   └── annotations/
└── RefSegRS/                   # RefSegRS dataset
    ├── images/
    ├── masks/
    ├── output_phrase_train.txt
    ├── output_phrase_val.txt
    └── output_phrase_test.txt
```

## Training

### Train on RRSIS-D
```bash
python train.py --dataset rrsis_d --data_root ./data --sam3_ckpt ./pre-trained-weights/sam3.pt
```

### Train on RRSIS-HR
```bash
python train.py --dataset rrsis_hr --data_root ./data --sam3_ckpt ./pre-trained-weights/sam3.pt
```

### Train on RefSegRS
```bash
python train.py --dataset refsegrs --data_root ./data --sam3_ckpt ./pre-trained-weights/sam3.pt
```

### Or use the launch script:
```bash
bash fine.sh rrsis_d ./data
bash fine.sh rrsis_hr ./data
bash fine.sh refsegrs ./data
```

### Key Training Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--image_size` | 504 | Input size (divisible by 14) |
| `--lora_rank` | 16 | LoRA adapter rank |
| `--batch_size` | 2 | Per-GPU batch size |
| `--grad_accum_steps` | 4 | Gradient accumulation (effective bs=8) |
| `--epochs` | 40 | Training epochs |
| `--lr` | 5e-5 | Learning rate |
| `--fp16` | True | Mixed precision training |

## Evaluation

```bash
python test.py --dataset rrsis_d --split test --resume ./output/rrsis_d_sam3/best_model.pth
```

### Output Metrics
- **mIoU**: Mean Intersection over Union
- **oIoU**: Overall IoU
- **P@0.5 - P@0.9**: Precision at IoU thresholds

## Project Structure
```
RRSIS_SAM3/
├── sam3/                  # SAM3 core (Meta's implementation)
│   ├── model/             # Encoder, decoder, ViT, text encoder
│   ├── sam/               # Prompt encoder, mask decoder, transformer
│   ├── train/             # Training utilities, losses, configs
│   └── assets/            # BPE vocabulary
├── lib/                   # RRSIS adaptation modules
│   ├── rrsis_sam3_model.py  # Main model wrapper
│   └── rs_adapters.py      # LoRA adapters
├── data/                  # Dataset loaders
│   └── dataset.py         # RRSIS-D, RRSIS-HR, RefSegRS
├── refer/                 # REFER API
├── loss/                  # Loss functions
├── configs/               # Training configs
├── pre-trained-weights/   # SAM3 checkpoint
├── train.py               # Training script
├── test.py                # Evaluation script
├── args.py                # CLI arguments
├── environment.yml        # Conda environment
└── requirements.txt       # Pip dependencies
```

## Citation
If you use this work, please cite:
```bibtex
@article{rrsis_sam3_2026,
    title={RRSIS-SAM3: Referring Remote Sensing Image Segmentation with SAM3},
    year={2026}
}
```

## License
This project uses SAM3 under Meta's license. See `sam3/LICENSE` for details.
