"""
RRSIS_SAM3 Training Script

Train SAM3 for Referring Remote Sensing Image Segmentation.
Supports: RRSIS-D, RRSIS-HR, RefSegRS datasets.

Usage:
    python train.py --dataset rrsis_d --data_root /path/to/data --sam3_ckpt ./pre-trained-weights/sam3.pt
"""

import os
import sys
import time
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.rrsis_sam3_model import RRSIS_SAM3
from lib.rs_adapters import get_trainable_params_summary

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU between predicted and target masks."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def get_optimizer(model, args):
    """
    Create optimizer with parameter groups for differentiated learning rates.

    Groups:
        1. LoRA parameters (backbone adapters) — lr_backbone
        2. Decoder/encoder parameters — lr_decoder
    """
    lora_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower():
            lora_params.append(param)
        elif any(x in name for x in ['transformer', 'segmentation_head', 'geometry_encoder', 'dot_prod']):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': lora_params, 'lr': args.lr_backbone, 'name': 'lora_adapters'},
        {'params': decoder_params, 'lr': args.lr_decoder, 'name': 'decoder'},
        {'params': other_params, 'lr': args.lr, 'name': 'other'},
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    for g in param_groups:
        print(f"  Param group '{g['name']}': {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler with warmup."""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


@torch.no_grad()
def validate(model, val_loader, device, epoch):
    """Run validation and compute metrics."""
    model.eval()
    iou_meter = AverageMeter()
    loss_meter = AverageMeter()

    for batch_idx, (images, masks, captions) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Handle eval mode captions (list of lists → flatten)
        if isinstance(captions[0], list):
            # For eval, use the first caption of each sample
            captions = [cap[0] for cap in captions]

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images, captions, masks)

        loss_meter.update(outputs['loss'].item(), images.size(0))
        iou = compute_iou(outputs['pred_masks'], masks)
        iou_meter.update(iou, images.size(0))

    print(f"  [Val] Epoch {epoch}: Loss={loss_meter.avg:.4f}, mIoU={iou_meter.avg:.4f}")
    return iou_meter.avg, loss_meter.avg


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    batch_time = AverageMeter()

    optimizer.zero_grad()
    end = time.time()

    for batch_idx, (images, masks, captions) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images, captions, masks)
            loss = outputs['loss'] / args.grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Metrics
        loss_meter.update(outputs['loss'].item(), images.size(0))
        with torch.no_grad():
            iou = compute_iou(outputs['pred_masks'], masks)
            iou_meter.update(iou, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"  [Train] Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss={loss_meter.avg:.4f} mIoU={iou_meter.avg:.4f} "
                  f"LR={lr_current:.2e} Time={batch_time.avg:.2f}s")

    return loss_meter.avg, iou_meter.avg


def main():
    args = get_args()
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  RRSIS_SAM3 Training")
    print(f"  Dataset: {args.dataset}")
    print(f"  Image Size: {args.image_size}×{args.image_size}")
    print(f"  LoRA Rank: {args.lora_rank}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size} × {args.grad_accum_steps} accum")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ====== Build Model ======
    print("Building RRSIS_SAM3 model...")
    model = RRSIS_SAM3(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        freeze_backbone=args.freeze_backbone,
        freeze_text_encoder=args.freeze_text_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = model.to(device)

    # ====== Build Datasets ======
    print("\nLoading datasets...")
    train_dataset = get_dataset(args, split='train', eval_mode=False)
    val_dataset = get_dataset(args, split='val', eval_mode=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Optimizer & Scheduler ======
    optimizer = get_optimizer(model, args)
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    scheduler = get_scheduler(optimizer, args, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ====== Resume ======
    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_iou = ckpt.get('best_iou', 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_iou={best_iou:.4f}")

    # ====== Output Directory ======
    os.makedirs(args.output_dir, exist_ok=True)

    # ====== Wandb ======
    if HAS_WANDB:
        wandb.init(
            project="RRSIS_SAM3",
            name=f"{args.dataset}_lr{args.lr}_lora{args.lora_rank}",
            config=vars(args),
        )

    # ====== Training Loop ======
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch + 1, args
        )

        # Validate
        val_iou, val_loss = validate(model, val_loader, device, epoch + 1)

        # Wandb logging
        if HAS_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Save best model
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args),
            }, save_path)
            print(f"  ★ New best model saved! mIoU={best_iou:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args),
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best mIoU: {best_iou:.4f}")
    print(f"{'='*60}")

    if HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
