import os
import csv
import cv2
from ultralytics import YOLO  # Replace with your YOLO library
import numpy as np

def parse_ground_truth(txt_file, target_class):
    ground_truth_boxes = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id, cx, cy, w, h = map(float, parts)
            if int(class_id) == target_class:
                ground_truth_boxes.append([cx, cy, w, h])
    return ground_truth_boxes

def convert_to_absolute(boxes, img_width, img_height):
    """
    Convert YOLO format boxes to absolute coordinates.
    """
    absolute_boxes = []
    for cx, cy, w, h in boxes:
        x_min = int((cx - w / 2) * img_width)
        y_min = int((cy - h / 2) * img_height)
        x_max = int((cx + w / 2) * img_width)
        y_max = int((cy + h / 2) * img_height)
        absolute_boxes.append([x_min, y_min, x_max, y_max])
    return absolute_boxes

def run_yolo_and_generate_csv(image_folder, output_csv, target_class):
    # Load YOLO model
    model = YOLO("/home/ajeet/Downloads/v8_nano_production_1.pt", verbose=True)

    # Open CSV file
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File", "Predicted_Confidences", "Predicted_Boxes", "Ground_Truth_Boxes"])

        for file_name in os.listdir(image_folder):
            if file_name.endswith('.jpg'):  # Assuming images are in .jpg format
                print(file_name)
                image_path = os.path.join(image_folder, file_name)
                txt_file = image_path.replace('.jpg', '.txt')
                if not os.path.exists(txt_file):
                    continue
                
                # Load image
                image = cv2.imread(image_path)
                img_height, img_width = image.shape[:2]

                # Run YOLO predictions
                image_path = os.path.join(image_folder, file_name)
                # results = model.predict(image_path, conf=0.25, classes=[2])
                # results = model.predict(image_path, conf=0.25, classes=[0])
                results = model.predict(image_path, conf=0.25, classes=[1])
                predictions = results[0].boxes  # Get bounding boxes

                # Filter predictions for the target class
                pred_confidences = []
                pred_boxes = []
                for box in predictions:
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    conf = round(box.conf[0].item(), 4)
                    class_id = int(box.cls[0])
                    if class_id == target_class:
                        pred_confidences.append(conf)
                        pred_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

                indices = cv2.dnn.NMSBoxes(pred_boxes, pred_confidences, score_threshold=0.25, nms_threshold=0.4)
                
                nms_pred_boxes = []
                nms_pred_confidences = []
                if len(indices) > 0:
                    for i in indices.flatten():
                        nms_pred_boxes.append(pred_boxes[i])
                        nms_pred_confidences.append(pred_confidences[i])

                # Parse ground truth boxes
                ground_truth_boxes = parse_ground_truth(txt_file, target_class)
                gt_boxes_abs = convert_to_absolute(ground_truth_boxes, img_width, img_height)

                # Write a single row for this image
                # writer.writerow([
                #     file_name,
                #     pred_confidences,
                #     pred_boxes,
                #     gt_boxes_abs
                # ])

                writer.writerow([
                    file_name,
                    nms_pred_confidences,
                    nms_pred_boxes,
                    gt_boxes_abs
                ])

# Example Usage
# image_folder = "/home/ajeet/codework/datasets/Training_Validation_Dataset_2024_Ajeet-20241107T105545Z-001/Training_Validation_Dataset_2024_Ajeet/test_2021/2021/test/test/optimal_thershold_on_5videos"
image_folder = "/home/ajeet/codework/datasets/Training_Validation_Dataset_2024_Ajeet-20241107T105545Z-001/Training_Validation_Dataset_2024_Ajeet/test_2021/2021/test/test/optimal_thershold_on_5videos"
output_csv = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/optimal_thershold/face+5videos_predictions_and_ground_truth.csv"

# run_yolo_and_generate_csv(image_folder, output_csv, target_class=2)
run_yolo_and_generate_csv(image_folder, output_csv, target_class=1)
