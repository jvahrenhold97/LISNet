import torch, os, csv
from tqdm import tqdm
from utilities import logging
import numpy as np
from model import serialization
from data import utils


def run_independent(device, config):
    """Run inference and evaluation on test set independently.
    
    Loads test data from config root directory and runs evaluation.
    Results saved to visualization and model directories.
    
    Args:
        device: PyTorch device (cuda or cpu).
        config (dict): Configuration dictionary with paths and parameters.
    
    Raises:
        RuntimeError: If test set has no valid labeled data.
    """
    curr_vis_dir = os.path.join(config.get("run_id"), "VIS")
    _, _, test_loader, _, _ = utils.fetch_data(
        config.get("root"),
        config.get("num_workers"),
        0,
        config.get("batch_size"),
        1.0,
        config.get("classes"),
        config.get("data_caching"),
    )
    if len(test_loader.dataset) == 0:
        raise RuntimeError("No valid test set with labels found!")
    run(test_loader, device, curr_vis_dir, config)


def run(test_loader, device, curr_vis_dir, config):
    """Evaluate model on test set and log results.
    
    Computes instance and center detection metrics, visualizes predictions,
    and saves results to CSV.
    
    Args:
        test_loader: Test data loader.
        device: PyTorch device.
        curr_vis_dir (str): Directory to save visualizations.
        config (dict): Configuration dictionary.
        curr_mod_dir (str): Directory containing model checkpoints.
        model (nn.Module or None): Pre-loaded model. If None, loads from checkpoint.
    """
    all_center_metrics = []
    all_inst_metrics = []
    model = serialization.load_for_inference(config.get("weights"), device)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Testing...", ncols=200)
        for i, batch in enumerate(pbar):
            batch_gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=False):
                pred, masks, scores, coords_list = model(batch_gpu["image"])
            if "centers" in batch_gpu:
                c_m = logging.compute_center_metrics_from_maps(
                    coords_list, batch_gpu["centers"], batch_gpu["offsets"], 
                    batch_gpu["valid_spatial_mask_s4"], 5.0, config.get("center_thresh"), config.get("max_queries")
                )
                all_center_metrics.append(c_m)
            if "instance" in batch_gpu:
                i_m = logging.compute_instance_metrics(
                    masks, scores, batch_gpu["instance"], batch_gpu["valid_spatial_mask_s2"], batch_gpu.get("visibility")
                )
                all_inst_metrics.append(i_m)
            if i == 0:
                vis_batch = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                logging.visualize_validation(
                    batch=vis_batch,
                    fields=pred, pred_masks=masks, pred_scores=scores, pred_coords_list=coords_list,
                    save_path=os.path.join(curr_vis_dir, f"testing_final_visualization.png"),
                    max_samples=4,
                )
    print("\n" + "=" * 110)
    print(f"{f'FINAL EVALUATION:':^110}")
    print("-" * 110)
    csv_results = {"run_id": config.get("run_id")}
    if all_center_metrics:
        total_c_tp = sum(m["TP"] for m in all_center_metrics)
        total_c_fp = sum(m["FP"] for m in all_center_metrics)
        total_c_fn = sum(m["FN"] for m in all_center_metrics)
        cp = total_c_tp / max(1, total_c_tp + total_c_fp)
        cr = total_c_tp / max(1, total_c_tp + total_c_fn)
        cf1 = 2 * cp * cr / max(1e-4, cp + cr)
        avg_loc_err = np.mean([m["loc_mean"] for m in all_center_metrics])
        csv_results.update({
            "center_precision": round(cp, 4),
            "center_recall": round(cr, 4),
            "center_f1": round(cf1, 4),
            "center_loc_err": round(float(avg_loc_err), 3)
        })
        print(f"[Center Detection] F1: {cf1:.4f} | Prec: {cp:.4f} | Rec: {cr:.4f} | Error: {avg_loc_err:.3f}px")
    if all_inst_metrics:
        all_pq_values = []
        for m in all_inst_metrics:
            if "PQ_batch_list" in m:
                all_pq_values.extend(m["PQ_batch_list"])
        final_pq = np.mean(all_pq_values) if all_pq_values else 0.0
        final_map = np.mean([m["mAP"] for m in all_inst_metrics])
        final_ap50 = np.mean([m["AP_50"] for m in all_inst_metrics])
        final_ap_s = np.mean([m["AP_S"] for m in all_inst_metrics])
        final_ap_m = np.mean([m["AP_M"] for m in all_inst_metrics])
        final_ap_l = np.mean([m["AP_L"] for m in all_inst_metrics])
        csv_results.update({
            "mAP": round(final_map, 4),
            "AP_50": round(final_ap50, 4),
            "AP_S": round(final_ap_s, 4),
            "AP_M": round(final_ap_m, 4),
            "AP_L": round(final_ap_l, 4),
            "PQ": round(final_pq, 4)
        })
        print(f"[Instance Seg]   PQ: {final_pq:.4f} | mAP: {final_map:.4f} | AP50: {final_ap50:.4f}")
        print(f"                 AP_S: {final_ap_s:.4f} | AP_M: {final_ap_m:.4f} | AP_L: {final_ap_l:.4f}")
    print("-" * 110)
    csv_path = os.path.join(curr_vis_dir, "test_results.csv")
    fieldnames = ["run_id", "center_precision", "center_recall", "center_f1", 
                  "center_loc_err", "mAP", "AP_50", "AP_S", "AP_M", "AP_L", "PQ"]
    try:
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            writer.writerow(csv_results)
        print(f"[SUCCESS] Results appended to {csv_path}")
    except Exception as e:
        print(f"[ERROR] CSV Write failed: {e}")