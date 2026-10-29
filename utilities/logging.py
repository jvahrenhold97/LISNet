import torch, os, cv2, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.funcs import find_peaks_with_voting
from scipy.ndimage import find_objects, binary_fill_holes


def prog_state(string):
    """Print a progress state message with horizontal separators.
    
    Args:
        string (str): The message to print.
    """
    hr = "-" * 120
    print(string)
    print(hr)


def log_configuration(args, device, run_id):
    """Log the active CLI configuration and training parameters.
    
    Args:
        args: Parsed command-line arguments with configuration settings.
        dataset_type (str): Type of dataset being used.
        device (str): Device being used for training (e.g., 'cuda:0').
        run_id (str): Unique identifier for the training run.
    """
    hr = "-" * 120
    print(hr)
    if getattr(args, "inference", False):
        mode_tag = "INFERENCE MODE"
    elif getattr(args, "finetune", False):
        mode_tag = "FINETUNING MODE"
    elif getattr(args, "label", False):
        mode_tag = "LABELING MODE"
    elif getattr(args, "ablate", False):
        mode_tag = "ABLATION MODE"
    elif getattr(args, "test", False):
        mode_tag = "TESTING MODE"
    else:
        mode_tag = "TRAINING MODE"
    dataset_tag = os.path.basename(args.root.strip("/")).upper()
    header = f"ACTIVE CLI CONFIGURATION | {mode_tag} | DATASET: {dataset_tag}"
    print(f"{header:^120}")
    print(hr)
    print(f"{'run':<60} |  {run_id}")
    arg_dict = vars(args)
    excluded_keywords = ["run_name", "root", "dir", "inference", "finetune", "label", "ablate", "test", "train"]
    sorted_keys = sorted(arg_dict.keys())
    for key in sorted_keys:
        if any(word in key.lower() for word in excluded_keywords):
            continue
        value = arg_dict[key]
        print(f"{key:<60} |  {value}")
    print(f"{'device':<60} |  {device}")
    print(hr)






























def plot_loss_curves(train_losses, val_losses, metrics, lr_history, save_path):
    """Plot training/validation losses, metrics, and learning rate schedule.
    
    Generates a 2-subplot figure with loss curves and composite metrics (mAP, center_f1 and PQ).
    Highlights the best model epoch and marks best validation loss epoch.
    
    Args:
        train_losses (dict): Dictionary with 'total' key containing training loss history.
        val_losses (dict): Dictionary with 'total' key containing validation loss history.
        metrics (dict): Dictionary containing 'mAP', 'center_f1' and 'PQ' metric histories.
        lr_history (list): Learning rate values over epochs.
        save_path (str): Path where the figure will be saved.
    """
    epochs = range(1, len(train_losses["total"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    train_sum = train_losses["total"]
    val_sum = val_losses["total"]
    ln1 = axs[0].plot(epochs, train_sum, label="Train Total", color="black", linewidth=1.5)
    ln2 = axs[0].plot(epochs, val_sum, label="Val Total", color="black", linestyle="--", linewidth=1.5)
    ax0_lr = axs[0].twinx()
    ln3 = ax0_lr.plot(epochs, lr_history, color="red", alpha=0.5, label="LR (Scale)")
    ax0_lr.set_ylabel("Learning Rate", color="black", fontsize=9)
    ax0_lr.tick_params(axis='y', labelcolor='black')
    best_val = float("inf")
    for i, val in enumerate(val_sum):
        if val < best_val:
            best_val = val
            axs[0].scatter(epochs[i], val, c="green", marker="+", s=60, zorder=6)
    axs[0].set_title("Total Loss & LR Schedule")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, alpha=0.3)
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc="upper right", fontsize=8)
    mAP_hist = np.array(metrics.get("mAP", [0.0]*len(epochs)))
    f1_hist = np.array(metrics.get("center_f1", [0.0]*len(epochs)))
    pq_hist = np.array(metrics.get("PQ", [0.0]*len(epochs)))
    axs[1].plot(epochs, mAP_hist, label="mAP", color="blue")
    axs[1].plot(epochs, pq_hist, label="PQ", color="purple")
    if len(mAP_hist) > 0 and len(pq_hist) > 0:
        weight_mAP = 0.25
        weight_f1 = 0.25
        weight_pq = 0.50
        composite_scores = (mAP_hist * weight_mAP) + (pq_hist * weight_pq) + (f1_hist * weight_f1)
        best_idx = np.argmax(composite_scores) 
        best_epoch = epochs[best_idx]
        axs[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.6, label=f"Best Model (Ep {best_epoch})")
        axs[1].scatter(best_epoch, mAP_hist[best_idx], color='blue', marker='*', s=100, zorder=5)
        axs[1].scatter(best_epoch, pq_hist[best_idx], color='purple', marker='*', s=100, zorder=5)
        axs[1].scatter(best_epoch, f1_hist[best_idx], color='yellow', marker='*', s=100, zorder=5)
    axs[1].set_title("Metrics")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def log_training_progress(train_avg_losses, val_avg_losses, criterion, metrics, metrics_interval, epoch, csv_path="training_log.csv"):
    """Log training progress to console and CSV file.
    
    Prints training/validation losses and metrics to console at evaluation epochs.
    Appends epoch data to a CSV log file.
    
    Args:
        train_avg_losses (dict): Dictionary with averaged training losses.
        val_avg_losses (dict): Dictionary with averaged validation losses.
        criterion: Loss criterion (unused but kept for API compatibility).
        metrics (dict or None): Dictionary containing computed metrics or None.
        metrics_interval (int): Interval for detailed metrics logging (0 for every epoch).
        epoch (int): Current epoch number.
        csv_path (str): Path to CSV file for logging. Defaults to 'training_log.csv'.
    """
    is_eval_epoch = (epoch == 1 or (metrics_interval > 0 and epoch % metrics_interval == 0))
    if is_eval_epoch:
        prog_state(
            f"[LOSSES] "
            f"TRAIN: {train_avg_losses['total']:.4f} | VAL: {val_avg_losses['total']:.4f}"
        )
        prog_state(
            f"[CENTER] "
            f"prec={metrics.get('center_prec', 0):.4f} | "
            f"rec={metrics.get('center_rec', 0):.4f} | "
            f"f1={metrics.get('center_f1', 0):.4f} | "
            f"loc_err={metrics.get('loc_mean', 0):.3f}px"
        )
        prog_state(
            f"[INSTNCE] "
            f"mAP@[.5:.95]={metrics.get('mAP', 0):.4f} | "
            f"AP_50={metrics.get('AP_50', 0):.4f} | "
            f"AP_S={metrics.get('AP_S', 0):.4f} | "
            f"AP_M={metrics.get('AP_M', 0):.4f} | "
            f"AP_L={metrics.get('AP_L', 0):.4f} | "
            f"PQ={metrics.get('PQ', 0):.4f}"
        )
    else:
        prog_state(f"EPOCH {epoch:03d} | TRAIN LOSS: {train_avg_losses['total']:.4f} | VAL LOSS: {val_avg_losses['total']:.4f}")
    file_exists = os.path.isfile(csv_path)
    if metrics is None:
        metrics = {}
    row_data = {
        "epoch": epoch,
        "train_loss_tot": train_avg_losses.get('total', 0),
        "val_loss_tot": val_avg_losses.get('total', 0),
        "center_prec": metrics.get('center_prec', ""),
        "center_rec": metrics.get('center_rec', ""),
        "center_f1": metrics.get('center_f1', ""),
        "loc_err_px": metrics.get('loc_mean', ""),
        "mAP_50_95": metrics.get('mAP', ""),
        "AP_50": metrics.get('AP_50', ""),
        "AP_S": metrics.get('AP_S', ""),
        "AP_M": metrics.get('AP_M', ""),
        "AP_L": metrics.get('AP_L', ""),
        "PQ": metrics.get('PQ', ""),
        "TP": metrics.get('TP', ""),
        "FP": metrics.get('FP', ""),
        "FN": metrics.get('FN', ""),
    }
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def calculate_coco_ap(matches, scores, total_gt):
    """Calculate COCO Average Precision (AP) at specified IoU threshold.
    
    Computes precision-recall curve and averages precision across 101 recall levels.
    
    Args:
        matches (list or array): Binary array indicating true positives (1) or false positives (0).
        scores (list or array): Confidence scores for predictions.
        total_gt (int): Total number of ground truth instances.
    
    Returns:
        float: Average Precision value between 0 and 1.
    """
    if not matches or total_gt == 0: return 0.0
    matches, scores = np.array(matches), np.array(scores)
    sort_idx = np.argsort(-scores)
    matches = matches[sort_idx]
    tp_cum = np.cumsum(matches)
    fp_cum = np.cumsum(1 - matches)
    rec = tp_cum / total_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-4)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = np.maximum(prec[i], prec[i + 1])
    recall_levels = np.linspace(0, 1, 101)
    precision_at_levels = np.zeros(101)
    for i, r in enumerate(recall_levels):
        relevant_prec = prec[rec >= r]
        if len(relevant_prec) > 0:
            precision_at_levels[i] = np.max(relevant_prec)
    return np.mean(precision_at_levels)


def get_size_category(area):
    """Categorize instance size as Small, Medium, or Large.
    
    Size thresholds: Small < 1024, Medium < 9216, Large >= 9216 pixels.
    
    Args:
        area (int or float): Instance area in pixels.
    
    Returns:
        str: Size category ('S', 'M', or 'L').
    """
    if area < 1024: return "S"
    elif area < 9216: return "M"
    else: return "L"


@torch.no_grad()
def compute_instance_metrics(masks, scores, gt_instance, valid_spatial_mask, visibility_map, score_thr=0.4, nms_iou_thr=0.5, mask_bin_thr=0.5, iou_range=(0.5, 0.95, 10)):
    """Compute instance segmentation metrics including mAP, PQ, and per-size-category AP."""
    iou_thresholds = np.linspace(*iou_range)
    ref_thr = iou_thresholds[0]
    cats = ["all", "S", "M", "L"]
    total_gt_instances = {c: 0 for c in cats}
    all_scores_at_thr = {c: {thr: [] for thr in iou_thresholds} for c in cats}
    all_matches_at_thr = {c: {thr: [] for thr in iou_thresholds} for c in cats}
    image_pqs = []
    global_tp = 0
    global_fp = 0
    global_fn = 0
    for b in range(len(masks)):
        vis = visibility_map[b].squeeze() if visibility_map is not None else None
        v_mask = valid_spatial_mask[b].squeeze()
        inst = gt_instance[b].squeeze()
        gt_ids = torch.unique(inst)
        gt_ids = gt_ids[gt_ids > 0]
        valid_gt_masks = []
        partial_gt_masks = []
        gt_cats = []
        for gt_id in gt_ids:
            gt_mask_bool = (inst == gt_id)
            if vis is not None:
                gt_vis_score = torch.unique(vis[gt_mask_bool]).max()
            else:
                gt_vis_score = 1.0
            if gt_vis_score > 0.5:
                valid_gt_masks.append(gt_mask_bool.float())
                gt_area = gt_mask_bool.sum()
                gt_cat = get_size_category(gt_area.item())
                gt_cats.append(gt_cat)
                total_gt_instances["all"] += 1
                total_gt_instances[gt_cat] += 1
            else:
                partial_gt_masks.append(gt_mask_bool)
        G = len(valid_gt_masks)
        if G > 0:
            gt_masks = torch.stack(valid_gt_masks, dim=0)
        else:
            gt_masks = None
        img_tp = 0
        img_fp = 0
        img_fn = 0
        img_iou_sum = 0.0
        if masks[b] is None or masks[b].numel() == 0:
            img_fn += G
            global_fn += G
            img_pq = 0.0 if G > 0 else 1.0
            image_pqs.append(img_pq)
            continue
        pred_scores = torch.sigmoid(scores[b])
        keep = pred_scores > score_thr
        if not keep.any():
            img_fn += G
            global_fn += G
            img_pq = 0.0 if G > 0 else 1.0
            image_pqs.append(img_pq)
            continue
        p_probs = torch.sigmoid(masks[b][keep])
        p_scores = pred_scores[keep]
        order = torch.argsort(p_scores, descending=True)
        p_probs, p_scores = p_probs[order], p_scores[order]
        keep_idx = []
        indices = torch.arange(len(p_probs), device=p_probs.device)
        while indices.numel() > 0:
            i = indices[0]
            keep_idx.append(i.item())
            if indices.numel() == 1: break
            curr = p_probs[i:i+1]
            rest = p_probs[indices[1:]]
            inter = (curr * rest * v_mask).sum((1, 2))
            union = (curr * v_mask).sum((1, 2)) + (rest * v_mask).sum((1, 2)) - inter
            iou_nms = inter / (union + 1e-4)
            indices = indices[1:][iou_nms <= nms_iou_thr]
        p_probs = p_probs[keep_idx]
        p_scores = p_scores[keep_idx]
        p_bin = (p_probs > mask_bin_thr).float()
        valid_area = (p_bin * v_mask.unsqueeze(0)).sum(dim=(1, 2))
        total_area = p_bin.sum(dim=(1, 2)) + 1e-4
        is_valid_pred = (valid_area / total_area) >= 0.5
        p_bin = p_bin[is_valid_pred]
        p_scores = p_scores[is_valid_pred]
        if len(partial_gt_masks) > 0 and len(p_bin) > 0:
            keep_preds = torch.ones(len(p_bin), dtype=torch.bool, device=p_bin.device)
            for q in range(len(p_bin)):
                pred_mask_bool = p_bin[q].bool()
                pred_area = pred_mask_bool.sum()
                for partial_gt in partial_gt_masks:
                    intersection = (partial_gt & pred_mask_bool).sum()
                    score = intersection / (pred_area + 1e-12)
                    if score > 0.5:
                        keep_preds[q] = False
                        break
            p_bin = p_bin[keep_preds]
            p_scores = p_scores[keep_preds]
        if len(p_scores) == 0:
            img_fn += G
            global_fn += G
            img_pq = 0.0 if G > 0 else 1.0
            image_pqs.append(img_pq)
            continue
        p_areas = p_bin.sum(dim=(1, 2))
        p_cats = [get_size_category(a.item()) for a in p_areas]
        if G == 0:
            num_fps = len(p_scores)
            img_fp += num_fps
            global_fp += num_fps
            for q, s in enumerate(p_scores):
                cat = p_cats[q]
                for c in ["all", cat]:
                    for thr in iou_thresholds: 
                        all_matches_at_thr[c][thr].append(0)
                        all_scores_at_thr[c][thr].append(float(s))
            image_pqs.append(0.0)
            continue
        p_bin_v = p_bin * v_mask.unsqueeze(0)
        gt_masks_v = gt_masks * v_mask.unsqueeze(0)
        inter = (p_bin_v[:, None] * gt_masks_v[None]).sum((2, 3))
        union = p_bin_v[:, None].sum((2, 3)) + gt_masks_v[None].sum((2, 3)) - inter
        iou_matrix = inter / (union + 1e-4)
        for thr in iou_thresholds:
            matched_gt = torch.zeros(G, dtype=torch.bool)
            for q in range(len(p_scores)):
                p_cat = p_cats[q]
                best_iou, best_gt_idx = iou_matrix[q].max(0)
                is_match = best_iou >= thr and not matched_gt[best_gt_idx]
                if is_match:
                    matched_gt[best_gt_idx] = True
                    gt_cat = gt_cats[best_gt_idx]
                    all_matches_at_thr["all"][thr].append(1)
                    all_scores_at_thr["all"][thr].append(float(p_scores[q]))
                    all_matches_at_thr[gt_cat][thr].append(1)
                    all_scores_at_thr[gt_cat][thr].append(float(p_scores[q]))
                    if p_cat != gt_cat:
                        all_matches_at_thr[p_cat][thr].append(0)
                        all_scores_at_thr[p_cat][thr].append(float(p_scores[q]))
                    if thr == ref_thr:
                        img_tp += 1
                        img_iou_sum += float(best_iou)
                else:
                    all_matches_at_thr["all"][thr].append(0)
                    all_scores_at_thr["all"][thr].append(float(p_scores[q]))
                    all_matches_at_thr[p_cat][thr].append(0)
                    all_scores_at_thr[p_cat][thr].append(float(p_scores[q]))
            if thr == ref_thr:
                batch_fps = (len(p_scores) - matched_gt.sum().item())
                batch_fns = (G - matched_gt.sum().item())
                img_fp += batch_fps
                img_fn += batch_fns
                global_tp += img_tp
                global_fp += img_fp
                global_fn += img_fn
        img_pq = (img_tp / max(img_tp + 0.5 * img_fp + 0.5 * img_fn, 1e-4)) * (img_iou_sum / max(img_tp, 1.0))
        image_pqs.append(img_pq)
    aps_all = [calculate_coco_ap(all_matches_at_thr["all"][t], all_scores_at_thr["all"][t], total_gt_instances["all"]) for t in iou_thresholds]
    aps_s = [calculate_coco_ap(all_matches_at_thr["S"][t], all_scores_at_thr["S"][t], total_gt_instances["S"]) for t in iou_thresholds]
    aps_m = [calculate_coco_ap(all_matches_at_thr["M"][t], all_scores_at_thr["M"][t], total_gt_instances["M"]) for t in iou_thresholds]
    aps_l = [calculate_coco_ap(all_matches_at_thr["L"][t], all_scores_at_thr["L"][t], total_gt_instances["L"]) for t in iou_thresholds]
    return {
        "mAP": float(np.mean(aps_all)),
        "AP_S": float(np.mean(aps_s)) if total_gt_instances["S"] > 0 else 0.0,
        "AP_M": float(np.mean(aps_m)) if total_gt_instances["M"] > 0 else 0.0,
        "AP_L": float(np.mean(aps_l)) if total_gt_instances["L"] > 0 else 0.0,
        f"AP_{int(ref_thr*100)}": float(aps_all[0]),
        "PQ_batch_list": image_pqs,
        "TP": float(global_tp),
        "FP": float(global_fp),
        "FN": float(global_fn),
    }


@torch.no_grad()
def compute_center_metrics_from_maps(coords, gt_centers, gt_offsets, valid_spatial_mask, match_radius, center_thresh, max_queries):
    """Compute center/keypoint detection metrics (precision, recall, F1, localization error).
    
    Matches predicted center coordinates with ground truth centers within a distance threshold.
    Only considers centers in valid spatial regions.
    
    Args:
        coords (list of torch.Tensor): Predicted center coordinates for each image.
        gt_centers (torch.Tensor): Ground truth center heatmaps.
        gt_offsets (torch.Tensor): Ground truth offset maps for center refinement.
        valid_spatial_mask (torch.Tensor): Validity mask for spatial regions (batch_size, 1, H, W).
        match_radius (float): Maximum distance for matching predicted to ground truth centers.
        center_thresh (float): Threshold for detecting centers in ground truth maps.
        max_queries (int): Maximum number of centers to detect.
    
    Returns:
        dict: Metrics including 'center_prec', 'center_rec', 'center_f1', 'loc_mean', 'TP', 'FP', 'FN'.
    """
    gt_coords, _ = find_peaks_with_voting(gt_centers, gt_offsets, center_thresh, max_queries)
    TP = FP = FN = 0
    loc_error_sum = 0.0
    for b, (pc, gc) in enumerate(zip(coords, gt_coords)):
        v_mask = valid_spatial_mask[b, 0]
        valid_pc = []
        for p in pc:
            y, x = int(p[0].item()), int(p[1].item())
            y = min(max(y, 0), v_mask.shape[0] - 1)
            x = min(max(x, 0), v_mask.shape[1] - 1)
            if v_mask[y, x] > 0.5:
                valid_pc.append(p)
        pc = torch.stack(valid_pc) if len(valid_pc) > 0 else torch.empty((0, 2), device=pc.device)
        num_pc = pc.shape[0]
        num_gt = gc.shape[0]
        if num_gt == 0:
            FP += num_pc
            continue
        if num_pc == 0:
            FN += num_gt
            continue
        dists = torch.cdist(pc.float(), gc.float())
        matched_gt_idx = set()
        matched_pc_idx = set()
        flattened_dists = dists.view(-1)
        sorted_indices = torch.argsort(flattened_dists)
        current_matches = 0
        for idx in sorted_indices:
            d_val = flattened_dists[idx].item()
            if d_val > match_radius:
                break
            r = idx // num_gt
            c = idx % num_gt
            if r.item() not in matched_pc_idx and c.item() not in matched_gt_idx:
                matched_pc_idx.add(r.item())
                matched_gt_idx.add(c.item())
                loc_error_sum += d_val
                current_matches += 1
            if current_matches == min(num_pc, num_gt):
                break
        TP += current_matches
        FP += num_pc - current_matches
        FN += num_gt - current_matches
    precision = TP / max(TP + FP, 1e-4)
    recall = TP / max(TP + FN, 1e-4)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-4)
    loc_mean = loc_error_sum / max(TP, 1)
    return {
        "center_prec": float(precision),
        "center_rec": float(recall),
        "center_f1": float(f1),
        "loc_mean": float(loc_mean),
        "TP": TP, 
        "FP": FP, 
        "FN": FN,
    }


def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a normalized image tensor to 0-255 uint8 format.
    
    Reverses standard ImageNet normalization and converts from CHW to HWC format.
    
    Args:
        img_tensor (torch.Tensor): Normalized image tensor (C, H, W) with values typically in [0, 1].
        mean (list): Mean values used for normalization. Defaults to ImageNet mean.
        std (list): Standard deviation values used for normalization. Defaults to ImageNet std.
    
    Returns:
        np.ndarray: Denormalized image as uint8 array in HWC format with values in [0, 255].
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def overlay_instances(image, inst_map, alpha=0.5):
    """Overlay colored instance masks on an image.
    
    Assigns random colors to each instance ID and blends with the original image.
    Instance ID 255 is rendered as dark gray (background/boundary).
    
    Args:
        image (np.ndarray): RGB image array (H, W, 3) with values in [0, 255].
        inst_map (np.ndarray or torch.Tensor): Instance map (H, W) with integer instance IDs.
        alpha (float): Blending factor for overlay. Defaults to 0.5 (50% transparency).
    
    Returns:
        np.ndarray: Blended image as uint8 array (H, W, 3) with instance colors overlaid.
    """
    h, w = image.shape[:2]
    if inst_map.shape[0] != h or inst_map.shape[1] != w:
        inst_map = cv2.resize(inst_map.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    colored = np.zeros_like(image, dtype=np.float32)
    ids = np.unique(inst_map)
    ids = ids[(ids > 0) & (ids != 255)]
    for inst_id in ids:
        mask = inst_map == inst_id
        color = np.random.randint(50, 255, size=3)
        colored[mask] = color
    blended = cv2.addWeighted(image.astype(np.float32), 1 - alpha, colored, alpha, 0)
    if 255 in np.unique(inst_map):
        blended[inst_map == 255] = [50, 50, 50]
    return blended.astype(np.uint8)


def postprocess_instance_map(inst_tensor):
    """Post-process instance map by filling holes within instances.
    
    Uses binary morphological operations to fill small holes in predicted instance masks.
    
    Args:
        inst_tensor (torch.Tensor): Instance map tensor (H, W) with integer instance IDs.
    
    Returns:
        torch.Tensor: Post-processed instance map on the same device as input.
    """
    inst_np = inst_tensor.cpu().numpy()
    slices = find_objects(inst_np)
    for i, slc in enumerate(slices):
        if slc is not None:
            inst_id = i + 1
            crop = inst_np[slc]
            leaf_mask = (crop == inst_id)
            filled_mask = binary_fill_holes(leaf_mask)
            new_pixels = filled_mask & ~leaf_mask
            crop[new_pixels & (crop == 0)] = inst_id
    return torch.from_numpy(inst_np).to(inst_tensor.device)


def masks_to_instance_map(mask_logits, scores, mask_thresh=0.01, score_thresh=0.4, fill_holes=True, debug=False):
    """Convert mask logits and scores to an instance segmentation map.
    
    Applies dynamic thresholding weighted by confidence scores, performs winner-takes-all
    assignment, and optionally fills holes in masks.
    
    Args:
        mask_logits (torch.Tensor or None): Mask logits (Q, H, W) where Q is number of queries.
        scores (torch.Tensor): Prediction scores (Q,) for each query.
        mask_thresh (float): Base threshold for mask binarization. Defaults to 0.01.
        score_thresh (float): Confidence threshold for keeping predictions. Defaults to 0.4.
        fill_holes (bool): Whether to fill holes in instance masks. Defaults to True.
        debug (bool): Print debug information. Defaults to False.
    
    Returns:
        torch.Tensor: Instance map (H, W) with integer instance IDs or zeros for background.
    """
    if mask_logits is None or mask_logits.numel() == 0:
        if debug: print("[DEBUG] Early exit: mask_logits is None or empty.")
        return torch.zeros(mask_logits.shape[-2:], dtype=torch.long, device=mask_logits.device)
    mask_probs = torch.sigmoid(mask_logits)
    score_probs = torch.sigmoid(scores)
    if debug:
        print("\n" + "="*40)
        print(f"[DEBUG] Total queries received: {scores.shape[0]}")
        print(f"[DEBUG] Raw Score Probs -> Max: {score_probs.max().item():.4f} | Min: {score_probs.min().item():.4f} | Mean: {score_probs.mean().item():.4f}")
    keep_q = score_probs > score_thresh 
    if debug:
        print(f"[DEBUG] Queries surviving score_thresh ({score_thresh}): {keep_q.sum().item()}")
    if not keep_q.any():
        if debug: print("[DEBUG] FATAL: All queries died at the score threshold. Returning empty mask.")
        return torch.zeros(mask_logits.shape[-2:], dtype=torch.long, device=mask_logits.device)
    valid_mask_probs = mask_probs[keep_q]
    valid_scores = score_probs[keep_q]
    if debug:
        max_mask_probs_per_query = valid_mask_probs.flatten(1).max(dim=1)[0]
        rounded_scores = [round(val, 3) for val in valid_scores.tolist()]
        rounded_max_masks = [round(val, 3) for val in max_mask_probs_per_query.tolist()]
        print(f"[DEBUG] Surviving Scores: {rounded_scores}")
        print(f"[DEBUG] Peak Mask Probabilities for these queries: {rounded_max_masks}")
    weighted_probs = valid_mask_probs * valid_scores[:, None, None]
    max_prob, idx = weighted_probs.max(dim=0)
    winning_scores = valid_scores[idx] 
    dynamic_thresh = mask_thresh * winning_scores
    keep_px = max_prob > dynamic_thresh
    inst = torch.zeros_like(idx, dtype=torch.long)
    inst[keep_px] = idx[keep_px] + 1
    if fill_holes:
        inst = postprocess_instance_map(inst)
    if debug:
        total_pixels = keep_px.numel()
        foreground_pixels = (inst > 0).sum().item() 
        bg_pixels = total_pixels - foreground_pixels
        unique_ids = torch.unique(inst)
        print(f"[DEBUG] Dynamic Threshold range used -> Max: {dynamic_thresh.max().item():.4f} | Min: {dynamic_thresh.min().item():.4f}")
        print(f"[DEBUG] Pixels assigned to Foreground: {foreground_pixels} | Background: {bg_pixels}")
        print(f"[DEBUG] Final unique instance IDs in map: {unique_ids.tolist()}")
        print("="*40 + "\n")
    return inst
    

@torch.no_grad()
def visualize_validation(batch, fields, pred_masks, pred_scores, pred_coords_list, save_path, max_samples=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Visualize validation results with ground truth and predictions side-by-side.
    
    Creates a figure comparing input RGB, ground truth instance masks, and predicted instance
    maps with center coordinates. Saves as a high-resolution PNG image.
    
    Args:
        batch (dict): Batch data containing 'image' and optionally 'instance' (ground truth).
        fields (dict): Dictionary containing 'centers' heatmap predictions.
        pred_masks (list of torch.Tensor): Predicted mask logits for each image.
        pred_scores (list of torch.Tensor): Prediction confidence scores.
        pred_coords_list (list of torch.Tensor): Predicted center coordinates for each image.
        save_path (str): Path where the visualization figure will be saved.
        max_samples (int): Maximum number of images to visualize. Defaults to 4.
        mean (list): Mean values for denormalization. Defaults to ImageNet mean.
        std (list): Standard deviation values for denormalization. Defaults to ImageNet std.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images = batch["image"]
    gt_instances = batch.get("instance", None) 
    B_orig, _, H_img, W_img = images.shape
    B = min(B_orig, max_samples)
    target_size = (H_img, W_img)
    pred_centers = torch.sigmoid(fields["centers"])
    pred_centers = F.interpolate(pred_centers, size=target_size, mode="bilinear", align_corners=False)
    cols = 3 
    fig, axs = plt.subplots(B, cols, figsize=(4 * cols, 4 * B))
    if B == 1: axs = axs[None, :]
    for i in range(B):
        img_rgb = denormalize_image(images[i], mean, std)
        axs[i, 0].imshow(img_rgb)
        axs[i, 0].set_title("Input RGB")
        axs[i, 0].axis("off")
        if gt_instances is not None:
            gt_map = gt_instances[i].cpu().numpy()
            gt_overlay = overlay_instances(img_rgb, gt_map)
            axs[i, 1].imshow(gt_overlay)
        else:
            axs[i, 1].text(0.5, 0.5, "GT Not Available", ha='center', va='center')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")
        pr_coords = pred_coords_list[i].cpu().numpy() if len(pred_coords_list[i]) > 0 else np.empty((0, 2))
        pr_inst_map = masks_to_instance_map(
            F.interpolate(pred_masks[i].unsqueeze(1), size=target_size, mode="bilinear", align_corners=False).squeeze(1), 
            pred_scores[i]
        )
        pr_overlay = overlay_instances(img_rgb, pr_inst_map.cpu().numpy())
        axs[i, 2].imshow(pr_overlay)
        if pr_coords.shape[0] > 0:
            scale_y = H_img / fields["centers"].shape[-2]
            scale_x = W_img / fields["centers"].shape[-1]
            pr_y, pr_x = pr_coords[:, 0] * scale_y, pr_coords[:, 1] * scale_x
            axs[i, 2].scatter(pr_x, pr_y, c="white", s=5, marker=".", linewidths=1.0, label="Pred")
        axs[i, 2].set_title("Predicted Instances")
        axs[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()