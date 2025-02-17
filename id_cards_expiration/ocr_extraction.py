import os
import random
import pandas as pd
from google.cloud import vision

# Set up Google Vision API client
client = vision.ImageAnnotatorClient.from_service_account_json('/home/ajeet/Downloads/promising-haiku-450113-b1-27fb9f6d749a.json')

def detect_text(image_path):
    """Detects text in the given image using Google Vision API."""
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    texts = response.text_annotations
    return texts[0].description if texts else ""

def process_images(folder_path, num_samples=200):
    """Randomly selects images, applies OCR, and stores results in a DataFrame."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Select 500 random images
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
    print(len(sampled_images))

    # Store results
    results = []
    count = 0 
    for image in sampled_images:
        image_path = os.path.join(folder_path, image)
        ocr_result = detect_text(image_path)
        results.append({
            "image_name": image,
            "folder_name": os.path.basename(folder_path),
            "ocr_results": ocr_result
        })
        count = count + 1
        print(count)

    return pd.DataFrame(results)

# Set the folder containing images
folder_path = "/home/ajeet/Downloads/Labelled_ID_Cards_Train_V1-20250207T131933Z-001/Labelled_ID_Cards_Train_V1"

# Process images and save results to a CSV file
df = process_images(folder_path)
df.to_csv("/home/ajeet/codework/visiontasks_microservices/expiry_date/datasets/Labelled_ID_Cards_Train_V1_ocr_results.csv", index=False)

print("OCR process completed. Results saved to 'ocr_results.csv'.")
