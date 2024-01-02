import numpy as np
import os
from sklearn.metrics import jaccard_score, precision_score, f1_score
from skimage.io import imread
import sys

def compute_metrics(prediction_dir, ground_truth_dir):
    prediction_files = set(os.listdir(prediction_dir))
    ground_truth_files = set(os.listdir(ground_truth_dir))

    # Get the common files between predictions and ground truths
    common_files = prediction_files.intersection(ground_truth_files)

    if len(common_files) == 0:
        raise ValueError("No matching files found between prediction masks and ground truth masks.")

    ious = []
    precisions = []
    f1_scores = []

    for common_file in common_files:
        pred_path = os.path.join(prediction_dir, common_file)
        gt_path = os.path.join(ground_truth_dir, common_file)

        prediction_mask = imread(pred_path, as_gray=True)
        ground_truth_mask = imread(gt_path, as_gray=True)

        # Check if both masks have the same shape
        if prediction_mask.shape != ground_truth_mask.shape:
            print(f"Skipping {common_file} due to shape mismatch. Prediction shape: {prediction_mask.shape}, Ground truth shape: {ground_truth_mask.shape}")
            continue

        # Convert masks to binary format (0 or 1)
        prediction_mask = (prediction_mask > 0).astype(np.uint8)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

        prediction_mask_flat = prediction_mask.ravel()
        ground_truth_mask_flat = ground_truth_mask.ravel()

        ious.append(jaccard_score(ground_truth_mask_flat, prediction_mask_flat))
        precisions.append(precision_score(ground_truth_mask_flat, prediction_mask_flat))
        f1_scores.append(f1_score(ground_truth_mask_flat, prediction_mask_flat))

    return np.mean(ious), np.mean(precisions), np.mean(f1_scores)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compute_metrics.py /path/to/predictions /path/to/ground_truths")
        sys.exit(1)

    prediction_dir = sys.argv[1]
    ground_truth_dir = sys.argv[2]

    average_iou, average_precision, average_f1 = compute_metrics(prediction_dir, ground_truth_dir)
    print(f"Average IOU: {average_iou:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average F1-score: {average_f1:.4f}")
