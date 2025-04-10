import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import multiprocessing as mp

def get_valid_image_pairs(pred_dir, gt_dir, depth_anything_dir):
    depth_files = glob(os.path.join(depth_anything_dir, "*.png"))
    valid_numbers = set()
    for f in depth_files:
        basename = os.path.basename(f)
        number = os.path.splitext(basename)[0]
        if number.isdigit(): 
            valid_numbers.add(number)
    
    valid_pairs = []
    pred_files = glob(os.path.join(pred_dir, "*.png"))
    
    for pred_path in pred_files:
        basename = os.path.basename(pred_path)
        number = os.path.splitext(basename)[0]
        
        if number in valid_numbers:  
            gt_path = os.path.join(gt_dir, basename)
            if os.path.exists(gt_path):
                valid_pairs.append((pred_path, gt_path))
    
    return sorted(valid_pairs)

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    return (mask > 127).astype(np.float32)

def compute_metrics_single(pred_mask, gt_mask):
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    
    # IoU
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-6)
    
    # F-beta (beta=0.3)
    beta = 0.3
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f_beta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-6)
    
    # MAE
    mae = np.abs(pred_mask - gt_mask).mean()
    
    # BER
    ber = 1 - 0.5 * (tp / (tp + fn + 1e-6) + tn / (tn + fp + 1e-6))
    
    return {
        'iou': float(iou),
        'f_beta': float(f_beta),
        'mae': float(mae),
        'ber': float(ber)
    }

def evaluate_batch(args):
    pred_path, gt_path = args
    try:
        pred_mask = load_mask(pred_path)
        gt_mask = load_mask(gt_path)
        return compute_metrics_single(pred_mask, gt_mask)
    except Exception as e:
        print(f"Error processing {pred_path}: {str(e)}")
        return None

def evaluate_segmentation(pred_dir, gt_dir, depth_anything_dir, num_workers=None):
    valid_pairs = get_valid_image_pairs(pred_dir, gt_dir, depth_anything_dir)
    
    if not valid_pairs:
        raise ValueError("No valid image pairs found")
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    total_metrics = {
        'iou': 0.0,
        'f_beta': 0.0,
        'mae': 0.0,
        'ber': 0.0
    }
    valid_count = 0
    
    print(f"Processing {len(valid_pairs)} valid images using {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(evaluate_batch, valid_pairs), total=len(valid_pairs)))
    
    for result in results:
        if result is not None:
            valid_count += 1
            for key in total_metrics:
                total_metrics[key] += result[key]
    
    if valid_count > 0:
        for key in total_metrics:
            total_metrics[key] /= valid_count
    
    return total_metrics

if __name__ == "__main__":
    pred_dir = "./GSD/released_gsd_results/GSD" 
    gt_dir = "./mask"          
    depth_anything_dir = "./depth-anything-large-hf"  
    
    metrics = evaluate_segmentation(pred_dir, gt_dir, depth_anything_dir)
    
    print("\nSegmentation Metrics:")
    print(f"Total valid images processed: {len(get_valid_image_pairs(pred_dir, gt_dir, depth_anything_dir))}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"F-beta: {metrics['f_beta']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"BER: {metrics['ber']:.4f}")