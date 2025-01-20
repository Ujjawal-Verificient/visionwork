import pandas as pd
from gliner import GLiNER
import re


def replace_email_ids(text, replacement_email="ajee@gmail.com"):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)

    if matches:
        print(f"Found email IDs: {', '.join(matches)}")
    updated_text = re.sub(email_pattern, replacement_email, text)
    return updated_text


model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

input_file_path = r"/home/ajeet/Downloads/copy_chat.xlsx" 
df = pd.read_excel(input_file_path)

updated_descriptions = []
count = 0
for description in df['Description']: 
    labels = ["person", "phone number"]
    entities = model.predict_entities(description, labels)

    name_replacements = ["Ajay", "Pradeep"]
    unique_names = [] 
    new_description = description  

    new_description = replace_email_ids(new_description, "ajee@gmail.com")
    

    for entity in entities:
        if entity["label"] == "person" and entity["text"] != "I":
            if entity["text"] not in unique_names:
                unique_names.append(entity["text"])
                print(entity["text"], "=>", entity["label"])
            replacement_name = name_replacements[unique_names.index(entity["text"])] if unique_names.index(entity["text"]) < len(name_replacements) else entity["text"]
            new_description = new_description.replace(entity["text"], replacement_name)
        elif entity["label"] == "phone number":
            if any(char.isdigit() for char in entity["text"]):
                print(entity["text"], "=>", entity["label"])
                new_description = new_description.replace(entity["text"], "12345678910")
        # elif entity["label"] == "mailid":
        #     print(entity["text"], "=>", entity["label"])
        #     new_description = new_description.replace(entity["text"], "ajee@gmail.com")
    
    new_description = new_description.replace("verificient", "abc")
    new_description = new_description.replace("Verificient", "abc")
    new_description = new_description.replace("1 (844) 753-2020", "12345678910")
    new_description = new_description.replace("1-844-753-2020", "12345678910")
    new_description = new_description.replace("18447532020", "12345678910")

    updated_descriptions.append(new_description)

    # person_printed = False
    # for entity in entities:
    #     if entity["label"] == "person" and not person_printed:
    #         print(unique_names)
    #         person_printed = True
        
    #     if entity["label"] == "phone number" or entity["label"] == "mailid":
    #         print(entity["text"], "=>", entity["label"])

    print('\n')
    print("No: ", count)
    print("Original Description:", description)
    print('\n')
    print("Updated Description:", new_description)
    count = count + 1 
    print("--------------------------------------------------------")

df['Updated Description'] = updated_descriptions

output_file_path = "/home/ajeet/Downloads/copy_chat_clean_2.xlsx"
with pd.ExcelWriter(output_file_path, mode="w", engine="openpyxl") as writer:
    df.to_excel(writer, index=False)

print(f"Updated file saved to: {output_file_path}")
