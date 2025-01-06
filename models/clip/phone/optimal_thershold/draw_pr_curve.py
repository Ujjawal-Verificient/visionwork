import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Boxes should be in [x_min, y_min, x_max, y_max] format.
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def calculate_pr_curve(csv_file, iou_threshold=0.4):
    data = pd.read_csv(csv_file)

    print(f"{'Threshold':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}{'TP':<10}{'FP':<10}{'Missing':<10}")
    print("-" * 40)
    # Lists to store true positives, false positives, and false negatives
    all_tp = []
    all_fp = []
    all_fn = []
    all_confidences = []

    total_ground_truth = 0
    for _, row in data.iterrows():
        if not row['Predicted_Confidences'] or not row['Predicted_Boxes']:
            continue

        # Parse confidences and boxes
        confidences = eval(row['Predicted_Confidences'])
        if((len(confidences)))> 1:
            print(confidences)
        pred_boxes = eval(row['Predicted_Boxes'])
        gt_boxes = eval(row['Ground_Truth_Boxes'])

        total_ground_truth += len(gt_boxes) 

        matched_gt = set()
        tp = 0
        fp = 0

        for conf, pred_box in zip(confidences, pred_boxes):
            match_found = False
            for i, gt_box in enumerate(gt_boxes):
                if i not in matched_gt and iou(pred_box, gt_box) >= iou_threshold:
                    match_found = True
                    matched_gt.add(i)
                    all_tp.append(1)  # Mark as TP
                    all_fp.append(0)
                    all_confidences.append(conf)
                    # tp += 1
                    break
            if not match_found:
                # fp += 1
                all_tp.append(0)  # Not a TP
                all_fp.append(1)  # Mark as FP
                all_confidences.append(conf)

    # Calculate Precision and Recall at different thresholds
    # thresholds = np.linspace(0.1, 1, 20)
    # thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1]
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 
     0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 
     0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
     0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
    precisions = []
    recalls = []

    for t in thresholds:
        tp = sum([tp for conf, tp in zip(all_confidences, all_tp) if conf >= t])

        fp = sum([fp for conf, fp in zip(all_confidences, all_fp) if conf >= t])
        fn = total_ground_truth- tp 

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

        print(f"{t:<10.2f}{precision:<10.2f}{recall:<10.2f}{f1:<10.2f}{tp:>10.2f}{fp:>10}{fn:>10}")
        

    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label='PR Curve', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.legend()
    plt.show()

# Example Usage
csv_file = '/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/optimal_thershold/predictions_and_ground_truth.csv'
calculate_pr_curve(csv_file)