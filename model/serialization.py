import os
import glob
import torch
from collections import defaultdict
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
from model.lisnet import LISNet
from utilities import logging


def build_optimizer_and_scheduler(
    model,
    state: str,
    base_lr: float = 5e-5,
    max_epochs: int = 150,
    factor: float = 1.0,
    cosine_factor: float = 0.75,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Constructs AdamW optimizer and the paper-aligned four-phase learning rate schedule."""
    main_lr = base_lr * factor
    absolute_minimum_lr = float(5e-7) if (base_lr == 5e-5 and factor == 1.0) else float(main_lr * 0.01)

    if state == "finetune":
        for param in model.backbone.parameters():
            param.requires_grad = False
        if hasattr(model.backbone, "stage4"):
            for param in model.backbone.stage4.parameters():
                param.requires_grad = True
        else:
            print("[WARNING] Backbone missing 'stage4'. Fully frozen.")
    else:
        for param in model.parameters():
            param.requires_grad = True

    spatial_params = []
    mask_dec_base_params = []
    for name, param in model.mask_decoder.named_parameters():
        if not param.requires_grad:
            continue
        if "spatial_head" in name or "mask_scale" in name:
            spatial_params.append(param)
        else:
            mask_dec_base_params.append(param)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    center_head_params = [p for p in model.center_head.parameters() if p.requires_grad]
    semantic_head_params = [p for p in model.semantic_head.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    geometry_head_params = [p for p in model.geometry_head.parameters() if p.requires_grad]

    raw_groups = [
        {
            "params": decoder_params + semantic_head_params + mask_dec_base_params,
            "lr": main_lr,
            "weight_decay": 1e-4,
        },
        {
            "params": backbone_params,
            "lr": main_lr * 0.2 if state == "finetune" else main_lr * 0.1,
            "weight_decay": 1e-4,
        },
        {
            "params": center_head_params,
            "lr": main_lr,
            "weight_decay": 1e-4,
        },
        {
            "params": spatial_params,
            "lr": main_lr * 1.5 if state == "finetune" else main_lr * 3.0,
            "weight_decay": 1e-4,
        },
        {
            "params": geometry_head_params,
            "lr": main_lr,
            "weight_decay": 1e-3,
        },
    ]

    param_groups = [g for g in raw_groups if len(g["params"]) > 0]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

    warmup_epochs = 5 if max_epochs >= 50 else max(1, int(round(max_epochs * (5.0 / 150.0))))
    stabilize_epochs = 10 if max_epochs >= 50 else max(1, int(round(max_epochs * (10.0 / 150.0))))
    stabilize_end = warmup_epochs + stabilize_epochs

    c_factor = 0.75 if cosine_factor is None else float(cosine_factor)
    cosine_duration = int(round(max_epochs * c_factor))
    cosine_end = min(max_epochs, stabilize_end + cosine_duration)
    actual_cosine_duration = max(1, cosine_end - stabilize_end)
    cooldown_epochs = max(0, max_epochs - cosine_end)

    warmup_start_factor = 0.1 if state == "finetune" else max(1e-4, absolute_minimum_lr / main_lr)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    stabilizing_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=stabilize_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=actual_cosine_duration,
        eta_min=float(absolute_minimum_lr),
    )

    cooldown_factor = max(1e-4, float(absolute_minimum_lr / main_lr))
    cooling_scheduler = ConstantLR(
        optimizer,
        factor=cooldown_factor,
        total_iters=max(1, cooldown_epochs),
    )

    if cooldown_epochs > 0:
        schedulers = [warmup_scheduler, stabilizing_scheduler, cosine_scheduler, cooling_scheduler]
        milestones = [warmup_epochs, stabilize_end, cosine_end]
    else:
        schedulers = [warmup_scheduler, stabilizing_scheduler, cosine_scheduler]
        milestones = [warmup_epochs, stabilize_end]

    scheduler = SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )
    return optimizer, scheduler


def sanitize_scheduler_eta_min(scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    """Recursively coerces any list-based eta_min fields in loaded schedulers into scalar floats."""
    sub_schedulers = getattr(scheduler, "_schedulers", [scheduler])
    for s in sub_schedulers:
        if hasattr(s, "eta_min"):
            if isinstance(s.eta_min, (list, tuple)):
                s.eta_min = float(s.eta_min[0])
            else:
                s.eta_min = float(s.eta_min)


def try_resume(model, optimizer, scheduler, criterion, run_dir, device, state, config):
    """Loads checkpoint state and sanitizes loaded scheduler structures against list-type minimums."""
    torch.serialization.add_safe_globals([defaultdict, list])
    ckpt_filename = "last.pt" if state == "train" else "last_finetune.pt"
    ckpt_path = os.path.join(run_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        return (1, float("inf"), 0, False, None, None, None, None)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    criterion.load_state_dict(ckpt["criterion"])

    sanitize_scheduler_eta_min(scheduler)

    logging.prog_state(f"[RESUME] Resuming existing session: {os.path.basename(run_dir)}")
    logging.prog_state(f"[RESUME] Continuing from epoch {ckpt['epoch']}")
    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_loss"],
        ckpt["val_loss_patience"],
        True,
        ckpt.get("train_hist", None),
        ckpt.get("val_hist", None),
        ckpt.get("metrics_hist", None),
        ckpt.get("lr_hist", None),
    )


def save_ckpt(path, model, optimizer, scheduler, criterion, epoch, best_val_loss, score_patience, config=None, train_hist=None, val_hist=None, metrics_hist=None, lr_hist=None):
    """Serializes complete training checkpoint states to disk."""
    sanitize_scheduler_eta_min(scheduler)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss_patience": score_patience,
            "train_hist": train_hist,
            "lr_hist": lr_hist,
            "val_hist": val_hist,
            "metrics_hist": metrics_hist,
            "criterion": criterion.state_dict(),
            "config": config,
        },
        path,
    )


def load_for_inference(ckpt_path, device):
    """Instantiates model architecture from configuration and restores weights for evaluation."""
    logging.prog_state(f"[INFERENCE] Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    backbone_type = "convnext_tiny" if saved_cfg.get("tiny_backbone", False) else saved_cfg.get("backbone", "convnext_small")

    model = LISNet(
        num_classes=saved_cfg.get("num_classes", 2),
        max_queries=saved_cfg.get("max_queries", 64),
        center_thresh=saved_cfg.get("center_thresh", 0.3),
        fpn_channels=saved_cfg.get("channels", 128),
        decoder_width=saved_cfg.get("decoder_width", 256),
        decoder_layers=saved_cfg.get("decoder_layers", 6),
        decoder_heads=saved_cfg.get("decoder_heads", 8),
        backbone=backbone_type,
        num_def_points=saved_cfg.get("num_deform_points", 32),
        enable_closing=saved_cfg.get("enable_closing", False),
        enable_ms_dilation=saved_cfg.get("enable_ms_dilation", False),
        disable_coordatt=saved_cfg.get("disable_coordatt", False),
        disable_dropblock=saved_cfg.get("disable_dropblock", False),
        enable_spatial_bias=saved_cfg.get("enable_spatial_bias", False),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"[INFO] Successfully loaded LISNet with {saved_cfg.get('num_classes', 2)} classes")
    return model


def load_for_finetuning(criterion, ckpt_path, device):
    """Initializes LISNet architecture and restores pretrained weights for fine-tuning."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] Fine-tuning checkpoint not found at: {ckpt_path}")

    logging.prog_state(f"[FINETUNE] Loading pre-trained weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    backbone_type = "convnext_tiny" if saved_cfg.get("tiny_backbone", False) else saved_cfg.get("backbone", "convnext_small")

    model = LISNet(
        num_classes=saved_cfg.get("num_classes", 2),
        max_queries=saved_cfg.get("max_queries", 64),
        center_thresh=saved_cfg.get("center_thresh", 0.3),
        fpn_channels=saved_cfg.get("channels", 128),
        decoder_width=saved_cfg.get("decoder_width", 256),
        decoder_layers=saved_cfg.get("decoder_layers", 6),
        decoder_heads=saved_cfg.get("decoder_heads", 8),
        backbone=backbone_type,
        num_def_points=saved_cfg.get("num_deform_points", 32),
        enable_closing=saved_cfg.get("enable_closing", False),
        enable_ms_dilation=saved_cfg.get("enable_ms_dilation", False),
        disable_coordatt=saved_cfg.get("disable_coordatt", False),
        disable_dropblock=saved_cfg.get("disable_dropblock", False),
        enable_spatial_bias=saved_cfg.get("enable_spatial_bias", False),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()
    if "criterion" in ckpt:
        criterion.load_state_dict(ckpt["criterion"])
    logging.prog_state("[FINETUNE] Successfully warmed up model from pre-training checkpoint.")
    return model