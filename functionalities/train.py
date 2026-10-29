import os, torch
import numpy as np
from tqdm import tqdm
from data import utils
from utilities import logging
from collections import defaultdict
from model import serialization
from model.lisnet import LISNet
from model.losses import LISNetLoss
from functionalities import test

def run(device, config, gp=0.75):
    """Run training loop for LISNet model.
    
    Initializes model, optimizer, scheduler, and data loaders based on configuration.
    Calls training loop and generates test visualizations with best model.
    
    Args:
        device: PyTorch device (cuda or cpu).
        config (dict): Configuration dictionary with all training hyperparameters.
    """
    lr_factor = 1.0
    state="train"
    train_loader, val_loader, test_loader, num_classes, _ = utils.fetch_data(
        config.get("root"),
        config.get("num_workers"),
        config.get("augment"),
        config.get("batch_size"),
        config.get("subset"),
        config.get("classes"),
        config.get("data_caching"),
    )
    logging.prog_state(f"DATA LOADED SUCCESSFULLY | TRAIN: {len(train_loader.dataset)} | VAL: {len(val_loader.dataset)} | TEST: {len(test_loader.dataset)} | CLASSES: {num_classes}")
    backbone = "convnext_tiny" if config.get("tiny_backbone") == True else "convnext_small"
    model = LISNet(
        config.get("classes"), 
        config.get("max_queries"), 
        config.get("center_thresh"), 
        config.get("channels"), 
        config.get("decoder_width"), 
        config.get("decoder_layers"), 
        config.get("decoder_heads"), 
        backbone,
        config.get("num_deform_points"),
        config.get("enable_closing"),
        config.get("enable_ms_dilation"),
        config.get("disable_coordatt"),
        config.get("disable_dropblock"),
        config.get("enable_spatial_bias"),
    ).to(device)
    logging.prog_state(f"[PARAMS] Trainable Parameters: {model.trainable_parameters():,}")
    criterion = LISNetLoss(
        config.get("w_center"),
        config.get("w_offset"),
        config.get("w_mask"),
        config.get("w_score"),
        config.get("w_semantic"),
    ).to(device)
    optimizer, scheduler = serialization.build_optimizer_and_scheduler(
        model, 
        state, 
        config.get("lr"), 
        config.get("max_epochs"),
        lr_factor,
        config.get("cosine_factor"),
    )
    train_loop(
        state,
        model, 
        criterion, 
        train_loader, 
        val_loader,
        test_loader,
        scheduler, 
        optimizer, 
        device, 
        config,
        gp,
    )


def train_loop(state, model, criterion, train_loader, val_loader, test_loader, scheduler, optimizer, device, config, gp):
    os.makedirs(os.path.join(config.get("run_id"), "VIS"), exist_ok=True)
    curr_vis_dir = os.path.join(config.get("run_id"), "VIS")
    os.makedirs(os.path.join(config.get("run_id"), "WEIGHTS"), exist_ok=True)
    curr_run_dir = os.path.join(config.get("run_id"), "WEIGHTS")
    train_hist = defaultdict(list)
    val_hist = defaultdict(list)
    metrics_history = defaultdict(list)
    lr_history = []
    grace_period = int((gp * config.get("cosine_factor", 0.75)) * config.get("max_epochs"))
    cos_epochs = int(config.get("cosine_factor", 0.75) * config.get("max_epochs"))
    logging.prog_state(f"[SCHEDULE] Training with a cosine schedule over {cos_epochs} epochs and a grace period of {grace_period} epochs for {config.get('max_epochs')} epochs.")
    (start_epoch, best_saved_metric, score_patience, resumed, loaded_t_hist, loaded_v_hist, loaded_m_hist, loaded_lr_hist) = serialization.try_resume(model, optimizer, scheduler, criterion, curr_run_dir, device, state, config)
    if resumed:
        train_hist = loaded_t_hist
        val_hist = loaded_v_hist
        lr_history = loaded_lr_hist
        metrics_history = loaded_m_hist if loaded_m_hist is not None else defaultdict(list)
        best_score = best_saved_metric 
        logging.prog_state(f"[RESUME] Epoch={start_epoch} | Best Score={best_score:.4f}")
    else:
        start_epoch = 1
        best_score = -1.0
        score_patience = 0
        logging.prog_state("[START] Starting fresh end-to-end training")
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(start_epoch, config.get("max_epochs") + 1):
        logging.prog_state(f"[CURRENT EPOCH] {epoch}/{config.get('max_epochs')}")
        logging.prog_state(f"[LEARNING RATE] {optimizer.param_groups[0]['lr']:.6e}")
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        train_hist, train_avg_losses = train_epoch(model, optimizer, criterion, train_loader, device, train_hist, scheduler, scaler)
        val_hist, val_avg_losses, metrics_dict = validate_epoch(model, epoch, criterion, val_loader, device, val_hist, curr_vis_dir, state, config)
        metrics_history["mAP"].append(metrics_dict["mAP"])
        metrics_history["center_f1"].append(metrics_dict["center_f1"])
        metrics_history["PQ"].append(metrics_dict["PQ"])
        logging.plot_loss_curves(train_hist, val_hist, metrics_history, lr_history, save_path=os.path.join(curr_vis_dir, f"{state}_loss_curve.png"))
        logging.log_training_progress(train_avg_losses, val_avg_losses, criterion, metrics_dict, config.get("m_interval", 1), epoch, os.path.join(curr_vis_dir, f"{state}_log.csv"))
        mAP = metrics_dict.get("mAP", 0.0) 
        f1 = metrics_dict.get("center_f1", 0.0) 
        pq = metrics_dict.get("PQ", 0.0)
        current_score = (pq * 0.5) + (f1 * 0.25) + (mAP * 0.25)
        improved = current_score > best_score
        if improved:
            logging.prog_state(
                f"[IMPROVED] "
                f"Score: {current_score:.4f} (mAP: {mAP:.2f} | PQ: {100*pq:.2f} | F1: {f1:.2f}) | "
                f"Prev Best: {best_score:.4f} | "
                f"Δ: {current_score - best_score:.4f}"
            )
            best_score = current_score
            score_patience = 0
            serialization.save_ckpt(
                path=os.path.join(curr_run_dir, "best.pt") if state=="train" else os.path.join(curr_run_dir, "best_finetune.pt"), 
                model=model, optimizer=optimizer, scheduler=scheduler, 
                criterion=criterion, epoch=epoch,
                best_val_loss=best_score, score_patience=score_patience,
                config=config, train_hist=train_hist, val_hist=val_hist,
                metrics_hist=metrics_history, lr_hist=lr_history,
            )
        else:
            if epoch <= grace_period:
                logging.prog_state(
                    f"[GRACE PERIOD] "
                    f"Current Score: {current_score:.4f} | "
                    f"Prev Best: {best_score:.4f} | "
                    f"Ignoring stagnation for {grace_period - epoch} more epochs."
                )
                best_score = max(best_score, current_score) 
                score_patience = 0
            else:
                score_patience += 1
                logging.prog_state(
                    f"[NO IMPROVEMENT] "
                    f"Current Score: {current_score:.4f} | "
                    f"Prev Best: {best_score:.4f} | "
                    f"Patience {score_patience}/{config.get('patience')}"
                )
        serialization.save_ckpt(
            path=os.path.join(curr_run_dir, "last.pt") if state=="train" else os.path.join(curr_run_dir, "last_finetune.pt"),
            model=model, optimizer=optimizer, scheduler=scheduler, 
            criterion=criterion, epoch=epoch,
            best_val_loss=best_score, score_patience=score_patience,
            config=config, train_hist=train_hist, val_hist=val_hist,
            metrics_hist=metrics_history, lr_hist=lr_history,
        )
        if score_patience >= config.get("patience"):
            logging.prog_state(f"[EARLY STOPPING] Score hasn't improved for {config.get('patience')} epochs.")
            break
    best_model = serialization.load_for_inference(os.path.join(curr_run_dir, "best.pt"), device) if state == "train" else serialization.load_for_inference(os.path.join(curr_run_dir, "best_finetune.pt"), device)
    test.run(test_loader, device, curr_vis_dir, config, curr_run_dir, best_model)


def train_epoch(model, optimizer, criterion, train_loader, device, train_loss_history, scheduler, scaler):
    """Execute one training epoch with mixed precision and gradient accumulation.
    
    Computes losses, updates model weights, and updates learning rate schedule.
    
    Args:
        model: LISNet model instance.
        optimizer: Optimizer instance.
        criterion: Loss function.
        train_loader: Training data loader.
        device: PyTorch device.
        train_loss_history (dict): Dictionary to accumulate losses.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision training.
    """
    train_batches = 0
    train_loss_meters = defaultdict(float)
    model.train()
    pbar = tqdm(train_loader, desc="Training", ncols=160)
    for i, batch in enumerate(pbar):       
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred, masks, scores, coords_list = model(batch["image"])
            losses = criterion(pred, batch)
            loss = losses["total"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if i % 5 == 0:
            torch.cuda.synchronize() 
        with torch.no_grad():
            for k, v in losses.items():
                train_loss_meters[k] += v.detach() 
            train_batches += 1
    train_avg_losses = {k: (v / train_batches).item() for k, v in train_loss_meters.items()}
    for k, v in train_avg_losses.items():
        train_loss_history[k].append(v)
    scheduler.step()
    return train_loss_history, train_avg_losses


def validate_epoch(model, epoch, criterion, val_loader, device, val_loss_history, vis_dir, state, config):
    """Execute one validation epoch and compute metrics.
    
    Evaluates model on validation set, computes instance and center detection metrics,
    and saves visualizations.
    
    Args:
        model: LISNet model instance in eval mode.
        epoch (int): Current epoch number.
        criterion: Loss function.
        val_loader: Validation data loader.
        device: PyTorch device.
        val_loss_history (dict): Dictionary to accumulate losses.
        vis_dir (str): Directory to save visualizations.
        state (str): Training state ('train' or 'finetune').
        config (dict): Configuration dictionary.
    
    Returns:
        dict: Computed metrics (mAP, PQ, precision, recall, F1, etc.).
    """
    val_loss_meters = defaultdict(float)
    all_inst_metrics, all_center_metrics = [], []
    val_batches = 0
    model.eval()
    pbar = tqdm(val_loader, desc="Validation", ncols=200)
    with torch.no_grad():
        for batch in pbar:
            val_batches += 1
            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=False):
                pred, masks, scores, coords_list = model(batch["image"])
                losses = criterion(pred, batch)
            for k, v in losses.items():
                val_loss_meters[k] += float(v.detach())
            if val_batches == 1:
                cpu_batch = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                logging.visualize_validation(
                    batch=cpu_batch,
                    fields=pred, 
                    pred_masks=masks, 
                    pred_scores=scores, 
                    pred_coords_list=coords_list,
                    save_path=os.path.join(vis_dir, config.get("run_id"), f"{state}_epoch_{epoch:03d}.png"),
                    max_samples=4,
                )
            calc_metrics = (epoch == 1) or (config.get("m_interval") > 0 and epoch % config.get("m_interval") == 0)
            if calc_metrics:
                c_m = logging.compute_center_metrics_from_maps(
                    coords_list, batch["centers"], batch["offsets"], 
                    batch["valid_spatial_mask_s4"], 5.0, config.get("center_thresh"), config.get("max_queries")
                )
                all_center_metrics.append(c_m)
                i_m = logging.compute_instance_metrics(
                    masks, scores, batch["instance"], batch["valid_spatial_mask_s2"], batch.get("visibility")
                )
                all_inst_metrics.append(i_m)
    val_avg_losses = {k: v / max(1, val_batches) for k, v in val_loss_meters.items()}
    for k, v in val_avg_losses.items():
        val_loss_history[k].append(v)
    total_tp = sum(m["TP"] for m in all_inst_metrics) if all_inst_metrics else 0
    total_fp = sum(m["FP"] for m in all_inst_metrics) if all_inst_metrics else 0
    total_fn = sum(m["FN"] for m in all_inst_metrics) if all_inst_metrics else 0
    def get_avg_inst(key):
        vals = []
        for m in all_inst_metrics:
            if key in m:
                if isinstance(m[key], list):
                    vals.extend(m[key])
                else:
                    vals.append(m[key])
        return float(np.mean(vals)) if vals else 0.0
    def get_avg_center(key):
        vals = []
        for m in all_center_metrics:
            if key in m:
                if isinstance(m[key], list):
                    vals.extend(m[key])
                else:
                    vals.append(m[key])
        return float(np.mean(vals)) if vals else 0.0
    total_c_tp = sum(m["TP"] for m in all_center_metrics) if all_center_metrics else 0
    total_c_fp = sum(m["FP"] for m in all_center_metrics) if all_center_metrics else 0
    total_c_fn = sum(m["FN"] for m in all_center_metrics) if all_center_metrics else 0
    c_prec = total_c_tp / max(1, total_c_tp + total_c_fp)
    c_rec = total_c_tp / max(1, total_c_tp + total_c_fn)
    c_f1 = 2 * c_prec * c_rec / max(1e-4, c_prec + c_rec)
    metrics_dict = {
        "mAP": get_avg_inst("mAP"),
        "AP_50": get_avg_inst("AP_50"),
        "AP_S": get_avg_inst("AP_S"),
        "AP_M": get_avg_inst("AP_M"),
        "AP_L": get_avg_inst("AP_L"),
        "PQ": get_avg_inst("PQ_batch_list"),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "center_prec": c_prec,
        "center_rec": c_rec,
        "center_f1": c_f1,
        "loc_mean": get_avg_center("loc_mean"),
    }
    return val_loss_history, val_avg_losses, metrics_dict