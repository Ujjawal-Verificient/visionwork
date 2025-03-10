import openai
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json

from openai import OpenAI

# client = OpenAI(api_key="sk-proj-k9YOSkj0LKfCcUj8A")

client = OpenAI(api_key="sk-proctorgpt-M2q")
import re

tags = {
    "403 Error": "When student gets 403 error on Proctortrack side apart from Cookies issue, not in live chat that will be covered in live chat issue.",
    "502 Bad Gateway": "When student gets 502 Bad Gateway on Proctortrack account, not in live chat that will be covered in live chat issue.",
    "Accommodation": "When student ask for any accomodation related query.",
    "CLEP - Account Creation": "The user facing issues while registering or creating account on the Proctortrack side",
    "Antivirus": "Issues where antivirus software blocks downloading, installing, or running Proctortrack app on user's machine.",
    "App Crash/App Freeze": "Issues where the Proctortrack application becomes unresponsive or freezes, or stops functioning correctly during operations, preventing users from progressing or completing their exams.",
    "System Crash/Shutdown": "Issues where the user's computer unexpectedly crashes or shuts down while running the Proctortrack application",
    "App Download": "Issues related to the downloading process of the Proctortrack application, including download failures, slow downloads, or incomplete downloads.",
    "Download page": "Issues related to accessing, downloading, or navigating the Proctortrack application's download page like such as broken links or page loading errors.",
    "App Launch - Windows": "Problems encountered when starting or launching the Proctortrack application on Windows.",
    "App Launch - MAC": "Problems encountered when starting or launching the Proctortrack application on macOS",
    "Auto Update Issue": "Problems related to the application's automatic update process, including update failures, the app becoming stuck during updates, or slow update performance due to internet connectivity issues.",
    "Blacklisted Apps": "Cases where the presence of unauthorized or interfering applications, files, or processes on the system prevents a user from accessing the exam or user unable to process further and requires steps such as closing applications",
    "Browser": "Issues related to using unsupported or incompatible web browsers when attempting to access the Proctortrack application.",
    "Browser Plugin": "Issue with Proctortrack browser plugin",
    "Proctortrack Exam Browser": "Difficulties encountered during the installation, launching, or operation of the dedicated Proctortrack Exam Browser, separate from standard web browsers.",
    "Cookies": "If the user is facing cookies related issues.",
    "CPU Usage": "If the app is crashing/Freezing due to high CPU usage",
    "CSV Issue": "This tag may use for exporting/importing/downloading the files from the instructor dashboard.",
    "Data Issues/Missing": "Issues involving incomplete, missing captured or loss of data.",
    "Data Purge": "When user wants to delete the Proctortrack account permanently",
    "DNS": "When the user is stuck on connecting step and we update the default DNS to resolve the issue",
    "Early Login": "When users attempt to access or start an exam before the officially scheduled start time but are prevented from doing so due to timing restrictions.",
    "Email Change": "When users requests to change the email address due to incorrect one.",
    "Exam Grades": "When user ask for exam scores  or performance results of their exam.",
    "Exam Lapsed": "Issues that arise when a user's exam session has lapsed or become inaccessible due to exceeding the scheduled time.",
    "Exam Password-Access Code": "Issues related to obtaining, entering, or verifying the password or access code required to initiate or access an exam.",
    "Exam Time Confirmation": "When a student inquires about or verifies the scheduled time or date.",
    "Exam-Scheduling": "When a student inquires about the exam schedule, requests a reschedule, or seeks guidance on scheduling an exam.",
    "Face Scan": "When student stuck on face scan process or unable to do the face scan or face scan not captured.",
    "ID Scan": "Applies when the issue involves problems with the ID scan, such as an unclear view or mismatches with the user's details.",
    "Instructor Dashboard": "When any instructor reports query related to instructor dashboard",
    "Live chat issue": "Issues encountered while using the proctor live chatbox.",
    "Live Video Feed": "Issues related to the live video stream during the proctoring session, like video lag, feed not loading.",
    "LMS - Login Issues Password": "Student unable to login to LMS platform.",
    "LMS Issues": "If any student is facing any issues from LMS side or any test page related issues can be tag under this tag",
    "Login Issues - Password": "When the student is facing login issues on a proctortrack (hosted) site.",
    "MacOS Upgrade": "When  there is a pending OSx update possibly affecting the working of PT app",
    "Microphone": "Any issues related to the Microphone",
    "Mobile App-Android": "Cover all the issues on android mobile regardless of the specific nature of the issue.",
    "Mobile App-IOS": "Cover all the issues on iPhone mobile or iPads regardless of the specific nature of the issue.",
    "Monitors/Display": "Issues related to multiple monitors or display configurations that interfere with exam processes, such as duplicate screens or external displays causing technical problems.",
    "Name Change": "Name change request by the user/instructors",
    "Network issue": "Issues related to internet connectivity or network stability.",
    "Onboarding": "Queries specifically about the process of completing onboarding exams or issues encountered during the onboarding exam. This does not include concerns about approval status or access issues after onboarding.",
    "Onboarding Status": "Inquiries related to the progress, approval, and completion of the onboarding process required to access actual exam.",
    "Onboarding Approval -Last min": "Issues that arise when the approval of the onboarding process occurs very close to the actual exam scheduled time.",
    "OS Restrictions -Windows": "If the PT app is getting obstructed due to OS-related permissions in Windows systems.",
    "OS Restrictions - MAC": "If the PT app is getting obstructed due to OS-related permissions in MAC systems.",
    "404 error": "If the student is getting a 404 error in procotortrack app.",
    "500 Error": "If the student is getting a 500 error in procotortrack app.",
    "504 Internal error": "If the student is getting a 504 error in procotortrack app.",
    "OTP": "Anything related to OTP",
    "Payment Issues": "All payment or refund related issues.",
    "QR Code": "If the learner is facing any issues related to PT app QR code scanning",
    "Room Scan - Tech Issues": "Technical difficulties encountered during the room scanning process required for exam setup, including room scan upload failures.",
    "Room Scan Instructions": "Inquires regarding the required steps for performing a room scan or concerns about compliance with room scan guidelines.",
    "Screencasting": "Issues related to screencasting devices.",
    "Server Downtime": "The ProctorTrack site or the student's university website is undergoing maintenance.",
    "Session Processing": "Issues in session reprocessing or sessions stuck in the processing stage.",
    "System Issues - Compatibility": "In case the student system doesn't meets the ProctorTrack app requirements",
    "Technical Requirements": "When a learner has concerns about the technical requirements for the proctortrack app.",
    "Test Configuration": "Queries related to test configuration.",
    "Test Issues - Date": "Queries related to test date or test expiry date.",
    "Test Issues - Reset / Resume": "Queries related to resuming the test or resetting the test attempt",
    "Test Issues-Registration": "Issues related to registering for exams on the Proctortrack dashboard.",
    "PT dashboard": "Learner's Proctortrack dashboard is not showing the scheduled test or other details as expected or dashboard is not loading.",
    "Test Submission": "When a learner encounters issues submitting the test or inquires about the test submission status or checks if their exam data has been received",
    "Upload Issues": "Issues related to upload proctoring data.",
    "Violations": "Issues related to flagged violations.",
    "Webcam": "Issues related to student's webcam functionality, including camera detection problems, and general webcam malfunctions.",
    "End Proctoring Button": "Issues related to 'End Proctoring' button.",
    "Institution Rescheduling": "Support requests related to changing or rescheduling exams or assessments through the educational institution, including modifying exam dates, handling missed exams, and adhering to institutional scheduling policies.",
    "App Connect": "Issues related to establishing or maintaining a connection with the Proctortrack application, including cases where the application is stuck on the welcome page.",
    "Test Page Redirect": "If the student is unable to be redirected to the exam page after proctoring started.",
    "Grant privilege page": "Support for problems encountered on the 'Grant Privileges' page",
    "Ticket ID": "If a student requests the ticket ID for their issue or communication with support.",
    "Permitted/Prohibited Items": "Queries about Approved or Restricted Items for the test.",
}

CLOSE_TAG_PAIRS = {
    "onboarding": ["onboardingstatus"], 
    "onboardingstatus": ["onboarding"],
    "Onboardingapprovallastmin": ["onboardingstatus"],
    "mobileappios":["roomscantechissues"],
    "appconnect": ["applaunchwindows"],
    "applaunchwindows": ["appconnect"],
    "appconnect": ["applaunchmac"],
    "earlylogin":["examtimeconfirmation"],
    "examtimeconfirmation": ["earlylogin"],
    "institutionrescheduling": ["examscheduling"],
    "blacklistedapps": ["antivirus"],
    "webcam": ["livevideofeed"],
    "browser": ["proctortrackexambrowser"],
    "proctortrackexambrowser": ["browser"],
    "appdownload": ["downloadpage"],
    "downloadpage": ["appdownload"], 
    "RoomScanTechIssues": ["roomscaninstructions"],
    "roomscaninstructions": ["RoomScanTechIssues"], 
}

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load 70 example tickets
example_tickets = pd.read_excel("/home/ajeet/Downloads/example_dataset.xlsx")  # Update the path
example_texts = example_tickets[['Description', 'Tags', 'Why this tag is applicable to this chat conversation']].to_dict(orient='records')
example_embeddings = model.encode([ex['Description'] for ex in example_texts], convert_to_tensor=True)

def normalize_tag(tag):
    return re.sub(r'[^a-zA-Z0-9]', '', tag.lower()) if isinstance(tag, str) else tag

def calculate_similarity(chat_text, tags):
    chat_embedding = model.encode(chat_text, convert_to_tensor=True)
    tag_descriptions = list(tags.values())
    tag_embeddings = model.encode(tag_descriptions, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(chat_embedding, tag_embeddings)[0]
    sorted_indices = similarities.argsort(descending=True)
    
    return sorted_indices, similarities

def select_tags(similarities, sorted_indices, tags, manual_tag):
    tag_keys = list(tags.keys())
    top_20_indices = sorted_indices[:20]
    top_20_tags = [tag_keys[i] for i in top_20_indices]
    
    normalized_manual_tag = normalize_tag(manual_tag)
    normalized_top_20_tags = [normalize_tag(tag) for tag in top_20_tags]
    
    if normalized_manual_tag not in normalized_top_20_tags:
        selected_tags = []
        for i in sorted_indices:
            selected_tags.append(tag_keys[i])
            if normalize_tag(tag_keys[i]) == normalized_manual_tag:
                break
    else:
        selected_tags = top_20_tags
    
    return selected_tags

def select_top_2_examples(chat_text):
    chat_embedding = model.encode(chat_text, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(chat_embedding, example_embeddings)[0]
    top_2_indices = similarities.argsort(descending=True)[:2]
    return [example_texts[i] for i in top_2_indices]

def parse_json_response(response_text):
    try:
        return json.loads(response_text)  # Try parsing normally
    except json.JSONDecodeError:
        # Handle cases where keys/values are not enclosed in double quotes
        response_text = re.sub(r'(\w+):', r'"\1":', response_text)  # Add double quotes to keys
        response_text = re.sub(r':\s*([^"{[\]}]+)(,|\})', r': "\1"\2', response_text)  # Add quotes around values
        return json.loads(response_text)  # Try parsing again

def openai_tagging(chat_text, selected_tags, tags):
    top_2_examples = select_top_2_examples(chat_text)
    tags_with_descriptions = {tag: tags[tag] for tag in selected_tags}

    # tags_with_descriptions = tags
    
    user_prompt_1 = f"""
    # CONTEXT #
    We are working on a proctoring system where users/learners interact with an AI chatbot for issue resolution. If unresolved, the conversation is forwarded to a support team member. 
    
    #############

    # OBJECTIVE #
    Yout task is to accurately tag the given 'CHAT CONVERSATION' based on the provided list of 'TAGS AND WHEN TO USE THEM' and the 'Examples'.
    Let's think step by step.
    
    # EXAMPLES #
    Example 1:
    Chat Conversation: {top_2_examples[0]['Description']}
    tag: {top_2_examples[0]['Tags']}
    reason: Let's think step by step. {top_2_examples[0]['Why this tag is applicable to this chat conversation']}
    reasoning: Let's think step by step. {top_2_examples[0]['Why this tag is applicable to this chat conversation']}
    
    Example 2:
    Chat Conversation: {top_2_examples[1]['Description']}
    tag: {top_2_examples[1]['Tags']}
    reason: Let's think step by step. {top_2_examples[1]['Why this tag is applicable to this chat conversation']}
    reasoning: Let's think step by step. {top_2_examples[0]['Why this tag is applicable to this chat conversation']}
    
    # CHAT CONVERSATION #
    {chat_text}

    #  TAGS AND WHEN TO USE THEM #
    {json.dumps(tags_with_descriptions, indent=2)}

    #############
    
    # RESPONSE #
    Do not include backticks or quotes in your response.
    Provide the output as a JSON object with the following structure:
    {{
        "tag": "<tag>",
        "reason": "<reason for selecting the tag>",
        "reasoning": "<Step-by-step explanation of how the tag was determined>"
    }}

    # IMPORTANT INSTRUCTIONS #
    - Often, users report an issue that may not accurately reflect the actual problem diagnosed by the support agent. In such cases, always prioritize tagging based on the final diagnosed issue by the support agent rather than the initially reported concern.

    """
    
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": user_prompt_1  
                    }
                ],
                temperature=0.7,
            )
    content = response.choices[0].message.content
    print(content)
    content = parse_json_response(content)
    return content

def openai_validate_tag(chat_text, manual_tag, predicted_tag, tags):
    top_2_examples = select_top_2_examples(chat_text)

    manual_tag_normalized = normalize_tag(manual_tag)
    predicted_tag_normalized = normalize_tag(predicted_tag)

    normalized_tags = {normalize_tag(key): value for key, value in tags.items()}

    manual_tag_description = normalized_tags.get(manual_tag_normalized, "No description available.")
    predicted_tag_description = normalized_tags.get(predicted_tag_normalized, "No description available.")

    print("manual_tag_description:", manual_tag_description)
    
    user_prompt_2 = f"""
    # CONTEXT #
    We are working on a proctoring system where users/learners interact with an AI chatbot for issue resolution. If unresolved, the conversation is forwarded to a support team member. The goal is to compare the correctness of two AI-predicted tags.

    #############

    # OBJECTIVE #
    Your task is to compare the two provided tags and determine which one best represents the 'CHAT CONVERSATION'. 
    Let's think step by step.

    # EXAMPLES #
    Example 1:
    Chat Conversation: {top_2_examples[0]['Description']}
    Tag: {top_2_examples[0]['Tags']}
    Reason: {top_2_examples[0]['Why this tag is applicable to this chat conversation']}
    
    Example 2:
    Chat Conversation: {top_2_examples[1]['Description']}
    Tag: {top_2_examples[0]['Tags']}
    Reason: {top_2_examples[0]['Why this tag is applicable to this chat conversation']}

    #############

    # CHAT CONVERSATION #
    {chat_text}

    # TAG 1 (Model 1 Prediction) #
    {manual_tag}
    
    # Cases where TAG 1 is applicable#
    {manual_tag_description}

    # TAG 2 (Model 2 Prediction) #
    {predicted_tag}
    
    # Cases where TAG 2 is applicable  #
    {predicted_tag_description}

    # RESPONSE #
    Do not include backticks or quotes in your response
    Compare the two tags along with their descriptions and decide which one best represents the chat conversation. 
    Respond in JSON format:
    {{
        "final_tag": "<Best tag after evaluation>",
        "justification": "<Explanation for why the final tag was chosen>",
        "description_update": "<If needed, rewrite tag descriptions to better capture similar cases>"
    }}

    # IMPORTANT INSTRUCTIONS #
    - Often, users report an issue that may not accurately reflect the actual problem diagnosed by the support agent. In such cases, always prioritize tagging based on the final diagnosed issue by the support agent rather than the initially reported concern.
    """
    
    # response = client.chat.completions.create(
    #             # model="o1-mini",
    #             model="gpt-4o",
    #             messages=[
    #                 {
    #                 "role": "user",
    #                 "content": user_prompt_2
    #                 }
    #             ],
    #             temperature=0.7,
    #         )
    response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {
                "role": "user",
                "content": user_prompt_2
                }
            ],
        )
    content = response.choices[0].message.content
    print(content)
    content = parse_json_response(content)
    return content

def process_chat_data(df, tags):
    results = []
    total_predictions = len(df)
    first_correct_predictions = 0
    final_correct_predictions = 0
    corrected_by_closness = 0
    
    count = 1
    for index, row in df.iterrows():
        try:
            chat_text = row['Updated Description']
            manual_tag = row['Tags']

            chat_text = """FAQ Bot Hello. How can I help you? 11:23 AM, 12th Feb CHRISTO Lischkoff Name update. 11:23 AM, 12th Feb FAQ Bot Oops! Couldn't get that, am a new bot in town and learning with time. You can try rephrasing your question. 11:23 AM, 12th Feb FAQ Bot Kindly re-phrase your query, so that we can suggest you the correct solution 11:23 AM, 12th Feb CHRISTO Lischkoff I need to update my name. 11:23 AM, 12th Feb FAQ Bot Please find the recommended article(s) 11:23 AM, 12th Feb FAQ Bot Was this helpful? 11:23 AM, 12th Feb CHRISTO Lischkoff No 11:23 AM, 12th Feb FAQ Bot Kindly provide your name 11:23 AM, 12th Feb CHRISTO Lischkoff Chris Lischkoff 11:24 AM, 12th Feb FAQ Bot Please provide your email address so that our support team can contact you if necessary. 11:24 AM, 12th Feb CHRISTO Lischkoff chris@lischkoff.ca 11:24 AM, 12th Feb FAQ Bot Please enter detailed description about your issue 11:24 AM, 12th Feb CHRISTO Lischkoff I had registered with a short version of my name. It's slightly different from my ID 11:24 AM, 12th Feb FAQ Bot Please wait while we transfer you to a support agent. 11:24 AM, 12th Feb Sayali Hi, This is Sayali from proctortrack support. How can I help you? 11:24 AM, 12th Feb CHRISTO Lischkoff I had Humber update my name to match my ID. As it didn't pass the onboarding yesterday. 11:25 AM, 12th Feb Sayali Please allow me a moment  11:26 AM, 12th Feb CHRISTO Lischkoff okay 11:26 AM, 12th Feb CHRISTO Lischkoff I think I just need to do the onboarding again 11:27 AM, 12th Feb Sayali Yes, Please retake your onboarding 11:29 AM, 12th Feb Sayali Is there anything else I can assist you with?  11:29 AM, 12th Feb Sayali Please don't hesitate to contact us if you still encounter any issues while taking the test. We will do our best to assist you. 11:30 AM, 12th Feb CHRISTO Lischkoff okay 11:30 AM, 12th Feb CHRISTO Lischkoff thanks 11:30 AM, 12th Feb Sayali  Thank you for contacting Verificient Support. Have a nice day ahead. 11:31 AM, 12th Feb Take to Team Inbox"""

            manual_tag = "Onboarding Status"

            sorted_indices, similarities = calculate_similarity(chat_text, tags)
            selected_tags = select_tags(similarities, sorted_indices, tags, manual_tag)
            
            predicted_response = openai_tagging(chat_text, selected_tags, tags)
            first_predicted_tag = predicted_response.get("tag", "")
            
            if normalize_tag(first_predicted_tag) == normalize_tag(manual_tag):
                first_correct_predictions += 1
                final_correct_predictions += 1
            
                results.append({
                    **row,
                    "first_predicted_tag": first_predicted_tag,
                    "final_predicted_tag": first_predicted_tag,
                    "description_update": None
                })
            else:
                try:
                    validation_response = openai_validate_tag(chat_text, manual_tag, first_predicted_tag, tags)
                    updated_tag = validation_response.get("final_tag", first_predicted_tag)
                    
                    print(updated_tag)
                    print(manual_tag)
                    if normalize_tag(updated_tag) == normalize_tag(manual_tag):
                        final_correct_predictions += 1
                    else:
                        if normalize_tag(updated_tag) in CLOSE_TAG_PAIRS.get(normalize_tag(manual_tag), []):
                            updated_tag = manual_tag
                            final_correct_predictions += 1

                    results.append({
                        **row, 
                        "first_predicted_tag": first_predicted_tag, 
                        "final_predicted_tag": updated_tag, 
                        "description_update": validation_response.get("description_update", None)
                    })

                except Exception as e:
                    print(f"Validation error at index {index}: {e}")
                    results.append({
                        **row, 
                        "first_predicted_tag": first_predicted_tag, 
                        "final_predicted_tag": first_predicted_tag,
                        "description_update": None
                    })

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            results.append({
                **row, 
                "first_predicted_tag": "ERROR", 
                "final_predicted_tag": "ERROR",
                "description_update": None
            })
    
        print(count)
        count = count + 1
        print("-"*40)
    
    first_accuracy = first_correct_predictions / total_predictions if total_predictions > 0 else 0
    final_accuracy = final_correct_predictions / total_predictions if total_predictions > 0 else 0

    print(first_correct_predictions)
    print(final_correct_predictions)
    print(f"First Prediction Accuracy: {first_accuracy * 100:.2f}%")
    print(f"Final Prediction Accuracy: {final_accuracy * 100:.2f}%")
    
    return pd.DataFrame(results)

df = pd.read_excel("/home/ajeet/Downloads/Clean_1stFeb_12Feb_Chat.xlsx")
processed_df = process_chat_data(df, tags)
processed_df.to_excel("6March_Clean_1stFeb_12Feb_Chat.xlsx", index=False)
