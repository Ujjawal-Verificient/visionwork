import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append("/home/ajeet/codework/ujjawal_github/visionwork/models/clip")

print("\n".join(sys.path))



import cv2
import os
import time
from PIL import Image
from modeling_yolo_finetuned import YOLOv8
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


print("Ajeet Singh")
print("yolo_clip_phone_detection")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

list_of_classes = ["a person", "a cell phone"]

def prediction(images):

    yolo_thershold = 0.001
    verify_yolo_by_clip_below_ther = 0.40
    detection = YOLOv8("/home/ajeet/Downloads/od_v8_nano_feb24_2.onnx" , yolo_thershold, 0.40)

    frames_result = []
    for image in images:
        final_label = "no_cell_phone"
        output_image, is_phone_detected, yolo_phone_score = detection.main(image)
        # print(f"{image}: {is_phone_detected} {yolo_phone_score} ")
        if is_phone_detected:
            final_label = list_of_classes[1]

        clip_prob = -1
        if 0.0001 < yolo_phone_score < verify_yolo_by_clip_below_ther:
            class_detected, clip_prob = clip_classifier(image)
            base_name = os.path.basename(image)
            if class_detected == list_of_classes[1]:
                final_label = list_of_classes[1]
                # print(f"cell_phone detected by clip for {base_name}::::: clip_result: {class_detected} {clip_prob} True")
            else:
                final_label = "no_cell_phone"
                # print(f"cell_phone_not detected by clip for {base_name}::::: clip_result: {class_detected} {clip_prob} False")

        frames_result.append({
                "frame_path": image,
                "final_label": final_label,
                "yolo_phone_score": yolo_phone_score,
                "clip_prob": clip_prob
            })
        
    print("-"*50)
    for result in frames_result:
        print(result)
        

def clip_classifier(image):
    frame_path = image
    image = Image.open(image)

    inputs = processor(text=list_of_classes, images=image, return_tensors="pt", padding=True)

    with torch.inference_mode():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image # image-text similarity score
    probs = logits_per_image.softmax(dim=1) # label probabilities
    max_prob_index = int(torch.argmax(probs))

    return list_of_classes[max_prob_index], probs



if __name__ == "__main__":

    # images = [os.path.join("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged") if filename.startswith("0_")]

    # images = [os.path.join("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged") ]

    # images = [os.path.join("/tmp/video_incidents_ajeet/ed2e9e4c-ec0e-437d-a13d-5a1b87340c44_20240911181755971334_20240911181756999046_0_merged_140332" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/ed2e9e4c-ec0e-437d-a13d-5a1b87340c44_20240911181755971334_20240911181756999046_0_merged_140332") if filename.startswith("0_")]

    # images = [os.path.join("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged") if filename.startswith("0_")]

    # images = [os.path.join("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged") ]

    # images = [os.path.join("/tmp/video_incidents_ajeet/b2045fbc-5af3-4b40-a166-072b90f803e5_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/b2045fbc-5af3-4b40-a166-072b90f803e5_merged") if filename.startswith("0_")]

    images = [os.path.join("/home/ajeet/codework/Cellphone_train/train" , filename) 
    for filename in os.listdir("/home/ajeet/codework/Cellphone_train/train")]

    # images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    images = images[:1000]
    print(len(images))
    prediction(images)
    