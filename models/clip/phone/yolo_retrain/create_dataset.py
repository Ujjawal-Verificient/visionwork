import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image
import random

def parse_xml_for_excluded_regions(xml_path):
    excluded_regions = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in {"person", "face", "hand"}:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            excluded_regions.append((xmin, ymin, xmax, ymax))

    return excluded_regions

def get_valid_phone_position(image, phone_width, phone_height, excluded_regions):
    img_width, img_height = image.size
    max_attempts = 100 

    for _ in range(max_attempts):
        xmin = random.randint(0, img_width - phone_width)
        ymin = random.randint(0, img_height - phone_height)
        xmax = xmin + phone_width
        ymax = ymin + phone_height

        overlaps = False
        for region in excluded_regions:
            rxmin, rymin, rxmax, rymax = region
            if not (xmax <= rxmin or xmin >= rxmax or ymax <= rymin or ymin >= rymax):
                overlaps = True
                break
        
        if not overlaps:
            return (xmin, ymin, xmax, ymax)
    
    raise ValueError("Could not find a valid placement for the phone.")

def save_updated_xml(output_xml_path, original_xml_path, phone_coords):
    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    phone_object = ET.Element("object")
    name = ET.SubElement(phone_object, "name")
    name.text = "phone"
    bndbox = ET.SubElement(phone_object, "bndbox")
    
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(phone_coords[0])
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(phone_coords[1])
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(phone_coords[2])
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(phone_coords[3])

    root.append(phone_object)

    tree.write(output_xml_path)

def process_image_with_coordinates(image1_path, xml_path, image2_path, phone_coords, output_xml_path):
    excluded_regions = parse_xml_for_excluded_regions(xml_path)

    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    # image2 = image2.resize((100, 100))

    xmin, ymin, xmax, ymax = phone_coords
    phone_crop = image2.crop((xmin, ymin, xmax, ymax))
    phone_width, phone_height = phone_crop.size

    try:
        target_coords = get_valid_phone_position(image1, phone_width, phone_height, excluded_regions)
    except ValueError:
        print(f"Could not find a valid position for phone in {image1_path}.")
        return image1 

    image1.paste(phone_crop, (target_coords[0], target_coords[1]))

    save_updated_xml(output_xml_path, xml_path, target_coords)

    return image1

def process_csvs_with_coordinates(csv1_path, csv2_path, output_dir):
    with open(csv2_path, newline='') as csvfile2:
        reader2 = csv.DictReader(csvfile2)
        phone_images_with_coords = [
            {
                "image_path": row["image_path"],
                "xmin": int(float(row["xmin"])),
                "ymin": int(float(row["ymin"])),
                "xmax": int(float(row["xmax"])),
                "ymax": int(float(row["ymax"]))
            }
            for row in reader2
        ]

    if not phone_images_with_coords:
        print("No phone images found in CSV2.")
        return

    phone_index = 0
    phone_count = len(phone_images_with_coords)

    with open(csv1_path, newline='') as csvfile1:
        reader1 = csv.DictReader(csvfile1)
        for row1 in reader1:
            image1_path = row1["Image Path"]
            phone_label = int(row1["label"])

            if phone_label == 0: 
                xml_path = image1_path.replace(".jpg", ".xml")
                output_xml_path = os.path.join(output_dir, os.path.basename(xml_path))

                phone_data = phone_images_with_coords[phone_index]
                image2_path = phone_data["image_path"]
                phone_coords = (
                    phone_data["xmin"],
                    phone_data["ymin"],
                    phone_data["xmax"],
                    phone_data["ymax"]
                )

                try:
                    modified_image = process_image_with_coordinates(
                        image1_path, xml_path, image2_path, phone_coords, output_xml_path
                    )

                    output_path = os.path.join(output_dir, os.path.basename(image1_path))
                    modified_image.save(output_path)
                    print(f"Processed and saved: {output_path}")

                    phone_index = (phone_index + 1) % phone_count
                except Exception as e:
                    print(f"Error processing {image1_path} with {image2_path}: {e}")

csv1_path = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/bb_data_phone_classifier_only/mv_snap_cts_feb.csv"  
csv2_path = "/home/ajeet/codework/visionworkajeet/models/my_visionwork/models/clip/phone/yolo_retrain_dataset/internet_downloaded.csv"
output_dir = "/home/ajeet/codework/detr_results_verify/output_images"
os.makedirs(output_dir, exist_ok=True)


process_csvs_with_coordinates(csv1_path, csv2_path, output_dir)
