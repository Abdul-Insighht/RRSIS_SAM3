"""
OT-Enhanced Loss for RRSIS_SAM3.

Combines:
  1. UOT matching between decoder queries and GT (from uotod)
  2. Classification loss weighted by OT transport plan
  3. Box regression loss (GIoU + L1) weighted by OT transport plan
  4. Mask segmentation loss (Dice + BCE) on the best-matched query

Reference:
    De Plaen et al., "Unbalanced Optimal Transport: A Unified Framework
    for Object Detection", CVPR 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from uotod.match import UnbalancedSinkhorn, BalancedSinkhorn
    from uotod.loss import GIoULoss, NegativeProbLoss
    HAS_UOTOD = True
except ImportError:
    HAS_UOTOD = False
    print("[WARNING] uotod not installed. Install with: pip install uotod")
    print("[WARNING] Falling back to basic Dice+BCE loss.")


class OTSegmentationLoss(nn.Module):
    """
    Combined OT detection + segmentation loss for RRSIS.

    For referring segmentation (single target per image):
    - Uses UOT to match N decoder queries to 1 GT object + background
    - Supervises ALL queries proportionally via OT matching weights
    - Applies mask loss (Dice + BCE) on the best-matched query

    Args:
        use_unbalanced: If True, use UnbalancedSinkhorn; else BalancedSinkhorn.
        background_cost: Cost threshold for matching to background class.
        reg_target: Regularization for GT constraint relaxation (unbalanced only).
        reg_pred: Regularization for prediction constraint (unbalanced only).
        mask_loss_weight: Weight for mask segmentation loss component.
        box_loss_weight: Weight for box regression loss component.
        cls_loss_weight: Weight for classification loss component.
        giou_weight: Weight for GIoU loss within box loss.
        l1_weight: Weight for L1 loss within box loss.
    """

    def __init__(
        self,
        use_unbalanced=True,
        background_cost=0.5,
        reg_target=1e-2,
        reg_pred=1.0,
        mask_loss_weight=5.0,
        box_loss_weight=2.0,
        cls_loss_weight=1.0,
        giou_weight=2.0,
        l1_weight=5.0,
    ):
        super().__init__()

        self.mask_loss_weight = mask_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight

        # Build OT matcher (only if uotod is available)
        self.matcher = None
        self.giou_loss_fn = None
        if HAS_UOTOD:
            if use_unbalanced:
                self.matcher = UnbalancedSinkhorn(
                    cls_match_module=NegativeProbLoss(reduction="none"),
                    loc_match_module=GIoULoss(reduction="none"),
                    background_cost=background_cost,
                    reg_target=reg_target,
                    reg_pred=reg_pred,
                )
            else:
                self.matcher = BalancedSinkhorn(
                    cls_match_module=NegativeProbLoss(reduction="none"),
                    loc_match_module=GIoULoss(reduction="none"),
                    background_cost=background_cost,
                )
            self.giou_loss_fn = GIoULoss(reduction="none")
            print("[OTLoss] UOT matching enabled")
        else:
            print("[OTLoss] Falling back to basic Dice+BCE loss (no uotod)")

    def forward(self, outputs, gt_masks, image_size):
        """
        Compute the combined OT detection + segmentation loss.

        Args:
            outputs: dict from RRSIS_SAM3 forward containing:
                - 'pred_masks': (B, 1, H, W) predicted mask logits
                - 'pred_logits': (B, N_queries, 1) confidence scores (optional)
                - 'pred_boxes': (B, N_queries, 4) cxcywh boxes (optional)
            gt_masks: (B, 1, H, W) ground truth binary masks.
            image_size: int, spatial size of images.

        Returns:
            total_loss: scalar tensor.
        """
        device = gt_masks.device

        # === 1. Mask Loss (Dice + BCE) — always computed ===
        mask_loss = self._compute_mask_loss(outputs['pred_masks'], gt_masks)

        # === 2. OT-based box + classification loss (if available) ===
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        if (self.matcher is not None
                and 'pred_boxes' in outputs
                and outputs['pred_boxes'] is not None
                and 'pred_logits' in outputs
                and outputs['pred_logits'] is not None):
            try:
                _box, _cls = self._compute_ot_losses(outputs, gt_masks, image_size)
                box_loss = _box
                cls_loss = _cls
            except Exception as e:
                # Graceful fallback — OT matching can fail on edge cases
                print(f"[OTLoss] OT matching failed ({e}), using mask loss only")

        # === 3. Total Loss ===
        total = (self.mask_loss_weight * mask_loss
                 + self.box_loss_weight * box_loss
                 + self.cls_loss_weight * cls_loss)

        return total

    def _compute_mask_loss(self, pred_masks, gt_masks):
        """Dice + BCE loss on predicted mask logits."""
        # Resize GT to match prediction if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_resized = F.interpolate(
                gt_masks.float(), pred_masks.shape[-2:], mode='nearest'
            )
        else:
            gt_resized = gt_masks.float()

        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_resized)

        # Dice loss
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_resized).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_resized.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice = dice.mean()

        return bce + dice

    def _compute_ot_losses(self, outputs, gt_masks, image_size):
        """
        Compute OT-based box regression + classification losses.

        Uses UOT/Balanced Sinkhorn to match all N decoder queries to
        the single GT target, then weights losses by the transport plan.
        """
        B = gt_masks.shape[0]
        device = gt_masks.device

        pred_boxes = outputs['pred_boxes']    # (B, N, 4)
        pred_logits = outputs['pred_logits']  # (B, N, 1)

        # Derive GT bounding box from mask
        gt_boxes = self._mask_to_box(gt_masks, image_size)  # (B, 1, 4)

        if gt_boxes is None:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # --- OT Matching ---
        input_dict = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }
        target_dict = {
            "labels": torch.zeros(B, 1, dtype=torch.long, device=device),
            "boxes": gt_boxes,
            "mask": torch.ones(B, 1, dtype=torch.bool, device=device),
        }

        matching = self.matcher(input_dict, target_dict)
        # matching shape: (B, N, 2) — [:,:,0] = foreground, [:,:,1] = background
        fg_weights = matching[:, :, 0]  # (B, N)

        # --- Weighted GIoU Loss ---
        gt_boxes_expanded = gt_boxes.expand_as(pred_boxes)  # (B, N, 4)
        giou_per_query = self.giou_loss_fn(
            pred_boxes.reshape(-1, 4),
            gt_boxes_expanded.reshape(-1, 4),
        ).view(B, -1)  # (B, N)
        giou_loss = (giou_per_query * fg_weights).sum() / fg_weights.sum().clamp(min=1)

        # --- Weighted L1 Loss ---
        l1_per_query = F.l1_loss(
            pred_boxes, gt_boxes_expanded, reduction='none'
        ).sum(dim=-1)  # (B, N)
        l1_loss = (l1_per_query * fg_weights).sum() / fg_weights.sum().clamp(min=1)

        box_loss = self.giou_weight * giou_loss + self.l1_weight * l1_loss

        # --- Classification Loss ---
        # Train logits to predict the OT foreground weight as soft target
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1),
            fg_weights.detach(),
            reduction='mean',
        )

        return box_loss, cls_loss

    @staticmethod
    def _mask_to_box(masks, image_size):
        """
        Convert binary masks to bounding boxes in normalized cxcywh format.

        Args:
            masks: (B, 1, H, W) binary masks.
            image_size: int, spatial dimension.

        Returns:
            boxes: (B, 1, 4) in cxcywh format, or None if all masks empty.
        """
        B = masks.shape[0]
        boxes = []
        for i in range(B):
            m = masks[i, 0]  # (H, W)
            if m.sum() == 0:
                # Empty mask → small centered box
                boxes.append(torch.tensor(
                    [0.5, 0.5, 0.01, 0.01], device=m.device, dtype=torch.float32
                ))
                continue
            rows = torch.any(m > 0.5, dim=1)
            cols = torch.any(m > 0.5, dim=0)
            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]
            h, w = m.shape
            cx = (cmin + cmax).float() / (2.0 * w)
            cy = (rmin + rmax).float() / (2.0 * h)
            bw = (cmax - cmin + 1).float() / w
            bh = (rmax - rmin + 1).float() / h
            boxes.append(torch.stack([cx, cy, bw.clamp(min=0.01), bh.clamp(min=0.01)]))

        return torch.stack(boxes).unsqueeze(1).to(masks.device)  # (B, 1, 4)
