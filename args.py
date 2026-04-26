import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RRSIS-SAM3: Referring Remote Sensing Image Segmentation with SAM3')

    # ====== Paths ======
    parser.add_argument('--data_root', type=str, default='./data/',
                        help='Root directory of datasets')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--sam3_ckpt', type=str, default='./pre-trained-weights/sam3.pt',
                        help='Path to SAM3 pretrained checkpoint')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume training from checkpoint')

    # ====== Dataset ======
    parser.add_argument('--dataset', type=str, default='refcoco',
                        choices=['refsegrs', 'rrsis_d', 'rrsis_hr'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='train',
                        help='Data split (train/val/test)')
    parser.add_argument('--splitBy', type=str, default='unc',
                        help='Split method')
    parser.add_argument('--max_tokens', type=int, default=32,
                        help='Maximum text token length for SAM3 tokenizer')

    # ====== Model ======
    parser.add_argument('--image_size', type=int, default=504,
                        help='Input image size (divisible by 14 for SAM3 ViT)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank for vision backbone adaptation')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze SAM3 ViT backbone (train only LoRA + decoder)')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True,
                        help='Freeze SAM3 text encoder')

    # ====== Training ======
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Base learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone (LoRA params)')
    parser.add_argument('--lr_decoder', type=float, default=5e-5,
                        help='Learning rate for decoder/seg head')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')

    # ====== Loss ======
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                        help='Weight for Cross-Entropy loss')
    parser.add_argument('--boundary_weight', type=float, default=0.2,
                        help='Weight for boundary loss')

    # ====== Optimization ======
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision (fp16/bf16)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing for memory savings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader workers')

    # ====== Distributed ======
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs')

    # ====== Evaluation ======
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Run evaluation only')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Save visualization of predictions')

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
