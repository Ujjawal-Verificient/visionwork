import csv
import openai
import openai
import pandas as pd

from openai import OpenAI

# client = OpenAI(api_key="sk-")

client = OpenAI(api_key="sk-q")

def generate_summary(chat_history, count ):
    my_text = f"""
    # CONTEXT # 
    We are working on a proctoring system where users interact with an AI chatbot for issue resolution. If unresolved, the conversation is forwarded to a support team member. The goal is to extract structured insights from the chat conversation, summarize the issue and the resolution provided.

    # OBJECTIVE #
    Extract insights from the chat conversation in a structured JSON format, including the user-reported issue and resolution provided.

    # STYLE # 
    Concise and technical, focusing on clarity and accuracy.

    # Tone # 
    Neutral and professional.

    # AUDIENCE # 
    Internal technical team analyzing support cases for reporting and improving user experience.

    # RESPONSE #
    The response must be in JSON format with fields: 
    {{
      "Summary of Issue": "",
      "Summary of Resolution Provided": ""
    }}

    # CHAT CONVERSATION # 
    {chat_history}
    """

    try:
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": my_text  
                }
            ]
            },
        ]
        )
        
        content = content = response.choices[0].message.content
        json_summary = eval(content.strip("```json"))
        print(f"No: {count}")
        print("Issue Description", chat_history)
        print('\n')
        print ("Issue Summary: ", json_summary.get("Summary of Issue", ""))
        print ("Resolution Provided: ", json_summary.get("Summary of Resolution Provided", ""))
        print("--------------------------------------------------------")
        return json_summary.get("Summary of Issue", ""), json_summary.get("Summary of Resolution Provided", "")
    except Exception as e:
        print(f"Error processing text: {chat_history}\nError: {e}")
        return "", ""

def process_excel(input_file, output_file):
    df = pd.read_excel(input_file)

    df['Summary of Issue'] = ""
    df['Summary of Resolution Provided'] = ""

    count = 1
    for index, row in df.iterrows():
        chat_text = row['Updated Description']
        
        if pd.notnull(chat_text):
            summary_of_issue, resolution_provided = generate_summary(chat_text, count)
            df.at[index, 'Summary of Issue'] = summary_of_issue
            df.at[index, 'Summary of Resolution Provided'] = resolution_provided

        count = count + 1

    df.to_excel(output_file, index=False)

input_file = r"/home/ajeet/Downloads/first_100_chats.xlsx"
output_file = r"/home/ajeet/Downloads/chat_summaires_100.xlsx"
process_excel(input_file, output_file)