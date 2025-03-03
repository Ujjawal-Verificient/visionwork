import csv
import openai
import json
import pandas as pd

from openai import OpenAI

# client = OpenAI(api_key="sk-proj)
client = OpenAI(api_key="sk-proctorgq")

def call_openai(system_prompt, user_prompt):
    """Calls OpenAI API with the given system and user prompt."""
    try:
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ],
        #     response_format={ "type": "json_object" },
        #     temperature=0.3
        # )
        response = client.chat.completions.create(
            model="o1-mini",  # Make sure the model supports chat completions
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
    return None

def call_openai_o1_mini(system_prompt, user_prompt):
    """Calls OpenAI API with the given system and user prompt."""
    try:
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
    return None

def generate_tagging(chat_history, tags_with_descriptions):
    """
    Generates the most appropriate tag for a given chat history.
    
    :param chat_history: str, The conversation history.
    :param tags_with_descriptions: dict, A dictionary of tags and their descriptions.
    :return: str, The most appropriate tag.
    """
    try:
        # 🟢 **First API Call - Select Top 3 Tags**
        system_prompt_1 = "You are an expert support ticket classifier. Your task is to analyze user conversations and select the top 3 most relevant tags based on given tags and their Descriptions."
        
        user_prompt_1 = f"""
            # CONTEXT # 
        We are working on a proctoring system where users/learners interact with an AI chatbot for issue resolution. If unresolved, the conversation is forwarded to a support team member. The goal is to accurately tag the given CHAT CONVERSATION based on the provided list of tag description.


        # CHAT CONVERSATION #
        {chat_history}

        # TAGS DESCRIPTIONS #
        {json.dumps(tags_with_descriptions, indent=2)}

        # RESPONSE # 
        Provide the output as a JSON object with the following structure:
        {{
            "tag": "<tag>",
            "reason": "<reason for selecting the tag>",
            "reasoning": "<Step-by-step explanation of how the tag was determined>"
        }}
        """
        
        top_3_tags_response = call_openai_o1_mini(system_prompt_1, user_prompt_1)
        print(top_3_tags_response)
        # cleaned_response = eval(top_3_tags_response.strip("```json"))
        cleaned_json = top_3_tags_response.strip("```json").strip("```").strip()
        top_3_tags = json.loads(cleaned_json)

        # Extract tag names
        # selected_tags = {tag_info["tag"]: tags_with_descriptions[tag_info["tag"]] for tag_info in top_5_tags}
        # selected_tags = {tag_info["tag"]: tags_with_descriptions[tag_info["tag"]] for tag_info in top_3_tags["tags"]}
        selected_tags = {tag_info["tag"]: tags_with_descriptions.get(tag_info["tag"], "Tag description not found") for tag_info in top_3_tags}


        # 🟢 **Second API Call - Choose the Most Accurate Tag**
        system_prompt_2 = "You are an AI specialized in support ticket classification. Your task is to review a chat history and choose the single most appropriate tag from a given list of top 3 tags."

        user_prompt_2 = f"""
        You are verifying the work of another AI model that has analyzed the chat and selected 3 relevant tags with reasoning. Think step by step.

        # CHAT CONVERSATION #  
        {chat_history}

        # TOP 3 AI-SELECTED TAGS # 
        {json.dumps(top_3_tags, indent=2)}

        # TAGS DESCRIPTIONS # 
        {json.dumps(selected_tags, indent=2)}


        # INSTRUCTIONS # 
        - Analyze the chat.
        - Pick the most accurate tag.

        # RESPONSE # 
        Provide the output as a JSON object with the following structure:
        {
            { 
                "tag": "<selected tag>",
                "reason": "<why this tag was chosen over the others>",
                "reasoning": "<Step-by-step explanation of how the tag was determined>"
            }
        }

        """
        
        final_tag_response = call_openai(system_prompt_2, user_prompt_2)
        print(final_tag_response.strip("```json").strip("```").strip())
        final_tag = json.loads(final_tag_response.strip("```json").strip("```").strip())

        return final_tag["tag"]
    except Exception as e:
        print(e)

    return None

def evaluate_pipeline(xlsx_file, tags_with_descriptions):
    """
    Evaluates the tagging pipeline against the ground truth labels in an Excel (.xlsx) file.
    
    :param xlsx_file: str, Path to the Excel file containing ticket data.
    :param tags_with_descriptions: dict, A dictionary of tags and their descriptions.
    :return: None (Prints accuracy results)
    """
    
    # Load the Excel file
    df = pd.read_excel(xlsx_file)

    # Ensure the Excel file has necessary columns
    if "Description" not in df.columns or "Tags" not in df.columns:
        raise ValueError("Excel file must contain 'chat_history' and 'ground_truth' columns.")

    # # Generate predicted tags for each chat history
    # df["predicted_tag"] = df["Description"].apply(lambda x: generate_tagging(x, tags_with_descriptions))

    predicted_tags = []

    for desc in df["Description"]:
        predicted_tag = generate_tagging(desc, tags_with_descriptions)
        predicted_tags.append(predicted_tag)

    # Assign the predicted tags to a new column in the dataframe
    df["predicted_tag"] = predicted_tags

    # Calculate accuracy
    accuracy = accuracy_score(df["Tags"], df["predicted_tag"])
    
    print(f"Tagging Accuracy: {accuracy * 100:.2f}%")
    
    # Save results to an Excel file
    df.to_excel("tagging_results.xlsx", index=False)
    print("Results saved to tagging_results.xlsx")

tags_with_descriptions = {
    "403 Error": "When student gets 403 error on Proctortrack side apart from Cookies issue",
    "502 Bad Gateway": "When student gets 502 Bad Gateway on Proctortrack account",
    "Accommodation": "When student ask for any accomodation related query.",
    "CLEP - Account Creation": "The user facing issues while registering or creating account on the Proctortrack side",
    "Antivirus": "Issues where antivirus software blocks downloading, installing, or running Proctortrack.",
    "App Crash/App Freeze": "Issues where the Proctortrack application becomes unresponsive or stalls during operations, preventing users from progressing or completing their exams.",
    "App Download": "Issues related to downloading the Proctortrack application.",
    "App Launch - Windows": "Problems encountered when starting or launching the Proctortrack application on Windows.",
    "App Launch - MAC": "Problems encountered when starting or launching the Proctortrack application on macOS",
    "Auto Update Issue": "Problems related to the application's automatic update process, including update failures, the app becoming stuck during updates, or slow update performance due to internet connectivity issues.",
    "Blacklisted Apps": "Some blacklisted apps or files or process are shown on system check and unable to process further.",
    "Browser": "Issues related to using unsupported or incompatible web browsers when accessing the Proctortrack application.",
    "Browser Plugin": "Issue with Proctortrack browser plugin",
    "Cookies": "If the user is facing cookies related issues.",
    "CPU Usage": "If the app is crashing/Freezing due to high CPU usage",
    "CSV Issue": "This tag may use for exporting/importing/downloading the files from the instructor dashboard.",
    "Data Issues/Missing": "Issues involving incomplete, missing captured or loss of data.",
    "Data Purge": "When user wants to delete the Proctortrack account permanently",
    "DNS": "When the user is stuck on connecting step and we update the default DNS to resolve the issue",
    "Download page": "Issues related to accessing, downloading, or navigating the Proctortrack application's download page",
    "Early Login": "When users login and unable to access the test before the scheduled test time.",
    "Email Change": "When users requests to change the email address due to incorrect one.",
    "Exam Grades": "When user ask for exam scores.",
    "Exam Lapsed": "Issues that arise when a user's exam session has lapsed or become inaccessible due to exceeding the scheduled time.",
    "Exam Password-Access Code": "Issues related to obtaining, entering, or verifying the password or access code required to initiate or access an exam.",
    "Exam Time Confirmation": "if any student contacts to check the exam scheduled time or date.",
    "Exam-Scheduling": "When a student inquires about the exam schedule, requests a reschedule, or seeks guidance on scheduling an exam.",
    "Face Scan": "When student stuck on face scan process or unable to do the face scan",
    "ID Scan": "When users are unable to do the ID scan or ID scan has unclear view after reviewing.",
    "Instructor Dashboard": "When any instructor reports query related to instructor dashboard",
    "Live chat issue": "Issues encountered in the proctor chatbox.",
    "Live Video Feed": "Issues encountered in the live video feeds.",
    "LMS - Login Issues Password": "Student unable to login to LMS platform.",
    "LMS Issues": "If any student is facing any issues from LMS side or any test page related issues can be tag under this tag",
    "Login Issues - Password": "When the student is facing login issues on a proctortrack (hosted) site.",
    "MacOS Upgrade": "When  there is a pending OSx update possibly affecting the working of PT app",
    "Microphone": "Any issues related to the Microphone",
    "Mobile App-Android": "Any Issues related to the Proctortrack application on android mobile.",
    "Mobile App-IOS": "Any Issues related to the Proctortrack application on Apple iOS devices.",
    "Monitors/Display": "issues related to multiple monitors or error related to monitors.",
    "Name Change": "Name change request by the learners/instructors",
    "Network issue": "Issues related to internet connectivity or network stability.",
    "Onboarding": "Queries related to Onboarding exam e.g how to take onboarding exam.",
    "Onboarding Status": "Inquiries related to the progress, approval, and completion of the onboarding process required to access actual exam.",
    "Onboarding Approval -Last min": "Issues that arise when the approval of the onboarding process occurs very close to the actual exam scheduled time.",
    "OS Restrictions -Windows": "If the PT app is getting obstructed due to OS-related permissions in Windows systems.",
    "OS Restrictions - MAC": "If the PT app is getting obstructed due to OS-related permissions in MAC systems.",
    "404 error": "If the student is getting a 404 error in procotortrack app.",
    "500 Error": "If the student is getting a 500 error in procotortrack app.",
    "504 Internal error": "If the student is getting a 504 error in procotortrack app.",
    "OTP": "Anything related to OTP",
    "Payment Issues": "All payment or refund related issues.",
    "PT dashboard": "Learner's Proctortrack dashboard is not showing the scheduled test or other details as expected or dashboard is not loading.",
    "QR Code": "If the learner is facing any issues related to PT app QR code scanning",
    "Room Scan -Tech Issues": "Technical difficulties encountered during the room scanning process required for exam setup, including room scan upload failures.",
    "Room Scan Instructions": "Inquiries about room scan instructions or issues with a student's room not meeting the room scan guidelines.",
    "Screencasting": "Issues related to screencasting devices.",
    "Server Downtime": "The ProctorTrack site or the student's university website is undergoing maintenance.",
    "Session Processing": "Issues in session reprocessing or sessions stuck in the processing stage.",
    "System Crash/Shutdown": "Issues where the user's computer unexpectedly crashes or shuts down while using the Proctortrack application",
    "System Issues - Compatibility": "In case the student system doesn't meets the ProctorTrack app requirements",
    "Technical Requirements": "When a learner has concerns about the technical requirements for the proctortrack app.",
    "Test Configuration": "Queries related to test configuration.",
    "Test Issues - Date": "Queries related to test date or test expiry date.",
    "Test Issues - Reset / Resume": "Queries related to resuming the test or resetting the test attempt",
    "Test Issues-Registration": "Issues related to registering for exams, or viewing exams on the Proctortrack dashboard.",
    "Test Submission": "When a learner encounters issues submitting the test on the test platform or inquires about the test submission status.",
    "Upload Issues": "Issues related to upload proctoring data.",
    "Violations": "Issues related to flagged violations.",
    "Webcam": "Issues related to student's webcam functionality, including camera detection problems, and general webcam malfunctions.",
    "End Proctoring Button": "Issues related to 'End Proctoring' button.",
    "Proctortrack Exam Browser": "Issues related to the installation, launching, and operation of the Proctortrack Exam Browser.",
    "Institution Rescheduling": "Support requests related to changing or rescheduling exams or assessments through the educational institution, including modifying exam dates, handling missed exams, and adhering to institutional scheduling policies.",
    "App Connect": "Issues related to establishing or maintaining a connection with the Proctortrack application.",
    "Test Page Redirect": "If the student is unable to be redirected to the exam page after proctoring started.",
    "Grant privilege page": "Support for problems encountered on the 'Grant Privileges' page",
    "Ticket ID": "If a student requests the ticket ID for their issue or communication with support.",
    "Permitted/Prohibited Items": "Queries about Approved or Restricted Items for the test.",
}

# Run evaluation on Excel file containing ticket data
evaluate_pipeline("/home/ajeet/Downloads/17_30Jan_dataset.xlsx", tags_with_descriptions)