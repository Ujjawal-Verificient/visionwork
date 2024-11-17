import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights
import sys


class MobileNetv2:

    def __init__(self):
        self.transform =transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def only_pad_image(self, image, target_size=(224, 224), pad_color=(0, 0, 0)):
        original_width, original_height = image.size

        padded_image = Image.new("RGB", target_size, pad_color)
        x_offset = (target_size[0] - original_width) // 2
        y_offset = (target_size[1] - original_height) // 2

        padded_image.paste(image, (x_offset, y_offset))

        return padded_image
    
    def getitem(self, img_path, box):
 
        image = Image.open(img_path).convert("RGB")
        left, top, width, height = box
        xmin = left
        ymin = top
        xmax = left + width
        ymax = top + height

        image = image.crop((xmin, ymin, xmax, ymax))
        image = self.only_pad_image(image)

        # plt.imshow(image)
        # plt.axis('off')  # Turn off axis
        # plt.show()

        if self.transform:
            image = self.transform(image)

        return image

class YOLOv8:

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box

        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        self.img = cv2.imread(self.input_image)

        self.img_height, self.img_width = self.img.shape[:2]

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.input_width, self.input_height))

        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))

        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []
        phone_detected = False
        phone_boxes = []

        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                # print(f"class_id: {class_id}")

                if class_id == 2:
                    phone_detected = True
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                    left = int((x - w / 2) * x_factor)
                    top = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # left = int(x)
                    # top = int(y)
                    # width = int(w)
                    # height = int(h)

                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)

            phone_boxes.append(box)

        return input_image , phone_detected, phone_boxes

    def main(self):
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        model_inputs = session.get_inputs()

        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        img_data = self.preprocess()
        outputs = session.run(None, {model_inputs[0].name: img_data})
        return self.postprocess(self.img, outputs)

def save_image(image, path, image_name):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, image_name), image)

if __name__ == "__main__":

    images = [os.path.join("/home/ajeet/codework/datasets/Cellphone_train/train" , filename) 
    for filename in os.listdir("/home/ajeet/codework/datasets/Cellphone_train/train") ]

    # images = [os.path.join("/home/ajeet/codework/datasets/yolo_mobileNet_test/" , filename) 
    # for filename in os.listdir("/home/ajeet/codework/datasets/yolo_mobileNet_test/") ]


    # images = images[:10]

    # images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    mobilenet_clssification = MobileNetv2()
    # num_classes = 1
    # model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    # checkpoint_path = '/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/finetuned_models/14Nov_MN_CP_0,01_to_0,0001_decrease_layers_80.pth' 
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.eval()

    checkpoint_path = '/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/finetuned_models/16Nov_Resnet18_layer_-1_e_80.pth' 
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    count = 1
    for image in images:
        detection = YOLOv8("/home/ajeet/Downloads/od_v8_nano_feb24_2.onnx",image, 0.50, 0.40)

        output_image, phone_detected, phone_boxes = detection.main()
        
        if phone_detected:
            print(f"{image}: phone_detected by yolo")
        else:
            print(f"{image}: phone_not_detected by yolo")

        save_path_with_phone = "/home/ajeet/codework/finetuned_yolo"
        image_name = os.path.basename(detection.input_image)
        save_image(output_image, save_path_with_phone, image_name)


        for box in phone_boxes:
            input = mobilenet_clssification.getitem(image, box)
            inputs = input.unsqueeze(0).to(device)
            outputs = model(inputs)

            outputs = torch.sigmoid(outputs)
            print(outputs)
            predicted = torch.round(outputs)

            if predicted == 1:
                print(f"phone verfied for {image} by mobileNet")
                break
            else:
                print(f"phone not verfied for {image} by mobileNet")

        print(count)
        count = count + 1
        print("\n")






        # Display the output image in a window
        # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        # cv2.imshow("Output", output_image)

        # # Wait for a key press to exit
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
