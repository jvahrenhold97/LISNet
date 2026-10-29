import torch, os, glob
from collections import defaultdict
from model.lisnet import LISNet
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
from utilities import logging


def build_optimizer_and_scheduler(
    model,
    state: str,
    base_lr: float = 5e-5,
    max_epochs: int = 150,
    factor: float = 1.0,
    cosine_span_ratio: float = 0.75,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Construct AdamW optimizer and the exact four-phase learning rate schedule from the paper."""
    main_lr = base_lr * factor
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
    warmup_epochs = 5 if max_epochs >= 50 else max(1, int(round(max_epochs * (5 / 150))))
    stabilize_epochs = 10 if max_epochs >= 50 else max(1, int(round(max_epochs * (10 / 150))))
    stabilize_end = warmup_epochs + stabilize_epochs
    cosine_epochs = int(round(max_epochs * cosine_span_ratio))
    cosine_end = min(max_epochs, stabilize_end + cosine_epochs)
    actual_cosine_duration = cosine_end - stabilize_end
    cooldown_epochs = max(0, max_epochs - cosine_end)
    warmup_start_factor = 0.1 if state == "finetune" else 0.01
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
    eta_mins = [group["lr"] * 0.01 for group in optimizer.param_groups]
    eta_min_arg = eta_mins[0] if len(eta_mins) == 1 else eta_mins
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=actual_cosine_duration,
        eta_min=eta_min_arg,
    )
    cooling_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=max(1, cooldown_epochs),
    )
    milestones = [
        warmup_epochs,
        stabilize_end,
        cosine_end,
    ]
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            stabilizing_scheduler,
            cosine_scheduler,
            cooling_scheduler,
        ],
        milestones=milestones,
    )
    return optimizer, scheduler


def try_resume(model, optimizer, scheduler, criterion, run_dir, device, state, config):
    """Attempt to resume training from a checkpoint.
    
    Loads model, optimizer, scheduler, and criterion state from the latest checkpoint.
    Returns default values if no checkpoint exists.
    
    Args:
        model: LISNet model instance.
        optimizer: PyTorch optimizer instance.
        scheduler: Learning rate scheduler instance.
        criterion: Loss function instance.
        model_dir (str): Directory containing model checkpoints.
        device: PyTorch device (cuda or cpu).
        state (str): Either 'train' or 'finetune' mode.
        config (dict): Configuration dictionary with 'dataset_name' and 'run_name'.
    
    Returns:
        tuple: (start_epoch, best_val_loss, val_loss_patience, is_resumed, train_hist, val_hist, metrics_hist, lr_hist)
    """
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
    """Save training checkpoint with model, optimizer, and history.
    
    Saves complete training state including model weights, optimizer state, and loss histories
    for resuming training or inference.
    
    Args:
        path (str): File path where checkpoint will be saved.
        model: LISNet model instance.
        optimizer: Optimizer with state.
        scheduler: Learning rate scheduler with state.
        criterion: Loss function with state.
        epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss seen so far.
        score_patience (int): Early stopping patience counter.
        config (dict or None): Training configuration dictionary.
        train_hist (dict or None): Training loss history.
        val_hist (dict or None): Validation loss history.
        metrics_hist (dict or None): Computed metrics history.
        lr_hist (list or None): Learning rate history per epoch.
    """
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
    """Load a trained LISNet model for inference.
    
    Reconstructs model architecture from saved configuration and loads checkpoint weights.
    Sets model to evaluation mode.
    
    Args:
        ckpt_path (str): Path to checkpoint file.
        device: PyTorch device (cuda or cpu).
    
    Returns:
        LISNet: Model in evaluation mode on the specified device.
    """
    logging.prog_state(f"[INFERENCE] Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if saved_cfg.get("tiny_backbone", False):
        backbone_type = "convnext_tiny"
    else:
        backbone_type = saved_cfg.get("backbone", "convnext_small")
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
        enable_spatial_bias=saved_cfg.get("enable_spatial_bias", False)
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"[INFO] Successfully loaded LISNet with {saved_cfg.get('num_classes', 2)} classes")
    return model


def load_for_finetuning(criterion, ckpt_path, device):
    """Load a pre-trained LISNet model for finetuning.
    
    Reconstructs model from saved configuration, loads checkpoint weights, and optionally
    loads criterion state. Sets model to training mode.
    
    Args:
        criterion: Loss function instance to load state into.
        ckpt_path (str): Path to pre-trained checkpoint file.
        device: PyTorch device (cuda or cpu).
    
    Returns:
        LISNet: Model in training mode on the specified device.
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] Fine-tuning checkpoint not found at: {ckpt_path}")
    logging.prog_state(f"[FINETUNE] Loading pre-trained weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if saved_cfg.get("tiny_backbone", False):
        backbone_type = "convnext_tiny"
    else:
        backbone_type = saved_cfg.get("backbone", "convnext_small")
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
        enable_spatial_bias=saved_cfg.get("enable_spatial_bias", False)
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()
    if "criterion" in ckpt:
        criterion.load_state_dict(ckpt["criterion"])
    logging.prog_state(f"[FINETUNE] Successfully warmed up model from pre-training checkpoint.")
    return model