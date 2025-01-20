import csv
import openai
import pandas as pd

from openai import OpenAI

# client = OpenAI(api_key="sk-")

client = OpenAI(api_key="sk-")

def generate_summary(chat_history, count):
#     prompt= f"""
#     # CONTEXT # 
#     abc

#     Provide the output as a JSON object with the following structure:
#     {{ "tag": "tag" }}
# """
    # Define the prompt
    prompt = f"""
    # CONTEXT # 
    We are working on a proctoring system where users/learner interact with an AI chatbot for issue resolution. If unresolved, the conversation is forwarded to a support team member. The goal is to tag the CHAT CONVERSATION based on the provided TAGS and DESCRIPTION for analysis and reporting purposes.

    # OBJECTIVE # 
    Tag the CHAT CONVERSATION with appropriate tag based on the issue discussed in the chat conversation and also provide a reason for selecting the tag.

    # STYLE # 
    Concise and structured, focusing on accurately tagging the chat conversation according to the provided TAGS and DESCRIPTION. Additionally, provide a explanation for why the selected tag applies.

    # AUDIENCE # 
    Internal technical team responsible for categorizing support cases and improving user experience.

    # RESPONSE # 
    Provide the output as a JSON object with the following structure:
    {{ 
        "tag": "<tag>", 
        "reason": "<reason for selecting this tag>"
    }}

    # TAGS and DESCRIPTION # 
    403 Error: When student gets 403 error on Proctortrack side apart from Cookies issue
    502 Bad Gateway: When student gets 502 Bad Gateway on Proctortrack account
    Accommodation: When learner ask for any accomodation related query like need to speak loudly while taking exam, need extra time, etc
    CLEP - Account Creation: The user facing issues while registering on the Proctortrack side(Mostly Clep students reaching out for this)
    Antivirus: When the antivirus was installed on student's system and it was restricting the Proctortrack app
    App Crash/App Freeze: When the Proctortrack app freeze during the process of starting or ending the exam or App files are crashed or currupted due to any reason/ App crash/Freeze/Can't END Proctoring.
    App Download: Proctortrack app is unable to download or after clicking on Download button it is giving error as XML/ When student is facing issues to downalod the PT app
    App Launch - Windows: Proctortrack app is not opening after the download in Windows Computer system
    App Launch - MAC: Proctortrack app is not opening after the download in MAC Computer system
    Auto Update Issue: While connecting the app it got auto update and had an issue in updating or stuck over there/ Auto update was too slow, due to internet issues.
    Blacklisted Apps: Some blacklisted apps are shown on system check and unable to process further from there even after force close
    Browser: Browser unable to connect to the application due to browser compatiblity issue
    Browser Plugin: Issue with browser plugin due to extensions
    Call Back - No Answer: User did not answer the call and no option to drop voice message
    CALLBACK REQUEST: Call back the user as per the requested date and time
    Careers: user contacted us and ask regarding the job or career in our organization
    Cookies: Need to allow 3rd party cookies on the browser if the user is unable to proceed further from browser permission step, etc
    CPU Usage: If the app is crashing/Freezing due to high CPU usage
    CSV Issue: This tag may use for exporting/importing/downloading the files from the instructor dashboard or support tool
    CV Disabled: Stucked on loading configuration and we disable the CV from account
    Data Issues/Missing: If there is data loss or data issues reported by student/instructor
    Data Purge: When user wants to delete the Proctortrack account permanently
    DNS: When the user is stuck on connecting step and we update the default DNS to resolve the issue
    Download page: If student have any issues with the download page or the downlaod page is blank or not loading
    Early Login: When users login to the account and unable to access the go to test button before the scheduled time
    Email Change: WHen users requests to change the email address due to incorrect one.
    Exam Grades: WHen user ask for exam scores
    Exam Lapsed: When exam status is showing as lapsed on the Proctortrack dashboard.
    Exam Lapsed- Room Scan: If a learner encounters a room scan issue and their exam is lapsed
    Exam Lapsed- Blacklisted Apps: If a learner encounters a blacklisted app issue and their exam has lapsed, please use this tag
    Exam Lapsed- Chatbox: If a learner encounters a chat box issue and their exam lapsed, please use this tag.
    Exam Lapsed- Live Feeds: If a learner encounters issues with live video feeds and their exam lapsed, please use this tag
    Exam Password-Access Code: When student have any exam password related queries
    Exam Time Confirmation: if any student contacts to check the exam scheduled time then please use this tag.
    Exam-Scheduling: When Student ask about the exam schedule/Reschedule requests or how to schedule an exam
    Face Scan: When student stuck on face scan process or unable to do the face scan
    General Queries: General queries regarding Proctortrack, breaks, calculator query, ect.
    Getting Started: We provide assistance to students who are unfamiliar with the process of taking the PT test, guiding them from the login stage to the commencement of the exam
    ID Scan: When users are unable to do the ID scan or ID scan has unclear view after reviewing
    Instructor Dashboard: When any instructor reports query related to instructor dashboard
    JIRA notification: Whenever the Janison team adds a comment on JIRA, a corresponding ticket will be generated on Freshdesk, and we will tag it under 'JIRA Notification'
    Live chat issue: When issues arise with the live chat box, we categorize them under 'Live Chat Box'
    Live Video Feed: When students Live video feeds are not visible to the proctor
    LMS - Login Issues Password: Student unable to login to LMS platform
    LMS Issues: If any student is facing any issues from LMS side or any test page related issues can be tag under this tag
    Login Issues - Password: When the learner is facing login issues on a hosted platform
    MacOS Upgrade: When  there is a pending OSx update possibly affecting the working of PT app
    Microphone: Any issues related to the Microphone
    Mobile App-Android: Any issues related to PT mobile app -android
    Mobile App-IOS: Any issues related to PT mobile app- IOS
    Monitors/Display: Will include issues related to multiple monitors or error related to monitors on PT app
    Name Change: Name change request by the learners/instructors
    Network issue: If a learner is facing network-related issues like a fluctuating network/restricted networks
    No action required: For irrelevant calls, in which no support is required related to PT
    Onboarding: When the learner is requesting assistance related to Onboarding/How to take onboarding
    Onboarding Status: when the learner has a query related to OB approval
    Onboarding Approval -Last min: When learners reach out to us for onboarding approval at the last minute or just before the actual exam
    One to One request: When an instructor or an admin requested special assistance for a learner
    OS Restrictions -Windows: If the PT app is getting obstructed due to OS-related permissions in Windows systems.
    OS Restrictions - MAC: If the PT app is getting obstructed due to OS-related permissions in MAC systems.
    404 error: If the learner is getting a 404 error
    500 Error: If the learner is getting a 500 error
    504 Internal error: If the learner is getting a 504 error on our hosted platforms
    OTP: Anything related to OTP
    Payment Issues: All payment related issued
    Privacy Concerns: If any student encounters issues related to privacy and needs to report them
    PT dashboard: Learners PT dashboard is not displaying the scheduled test or other details as expected
    QR Code: If the learner is facing any issues related to PT app QR code scanning
    Room Scan -Tech Issues: helping the learner at the room scan part or providing the required instructions about it
    Room Scan Delay: If Room scan is under review for long than the displayed time on screen
    Sales: Sales related queries will be tagged under this taag
    Screencasting: If casting devices connected
    Server Downtime: when the testing portal is down for a pre-informed activity/PT site is down
    Session Processing: In the cases , that requires reprocessing of sessions or sessions are stuck in processing 
    System Crash/Shutdown: When the learner reports the system crash issues
    System Issues - Compatibility: In case the system doesn't meets the PT app requirements
    Technical Requirements: When a learners concern is about the tech requirements for the PT app
    Test Configuration: to confirm/check the test configuration-related settings
    Test Issues - Date: the cases under Level 3 test's exam date be expired, students are advised to promptly reach out to their respective instructors or the University for further guidance and assistance.
    Test Issues - Reset / Resume: any concerns/queries related to resuming the test or resetting the test attempt
    Test Issues-Registration: If a learner is facing difficulties in registartion on their platform or despite being registered for courses, are unable to view tests on the Proctortrack dashboard
    Test Submission: When the learner is facing an issue in submitting the test on LMS
    Upload Issues: Instead of old and new sessions it could be just Upload issues
    Violations: any concerns/queries related to the violations marked/occured
    Webcam: all Webcam related concerns
    End Proctoring Button: once the students are clicking End proctoring button nothings happening
    App re-Install: If the issue is getting fixed after re-installing the app
    Proctortrack Exam Browser: If any student encounters issues with the Proctortrack exam browser, please categorize and tag those specific issues under 'Proctortrack Exam Browser'
    Customer Feedback: When we receive complaint or any feedback about the service from client/learners.
    Institution Rescheduling: Please use this tag for issues where students are advised to contact the institution for rescheduling, applicable for any L2, L3, L4, or DIY clients.
    Browser First Approach: This tag will be used in cases where the Proctortrack Exam browser is installed for a client, and learners mistakenly use the Proctortrack Exam browser despite it not being required.
    App Connect: This tag should be used when the Proctortrack app launches but fails to establish a connection.
    Test Page Redirect: Apply this tag when the test page does not open or launch after the proctoring session has begun.
    Grant privilege page: If the student is stuck on the 'Grant Privileges' page, we use this tag.
    Ticket ID: If a student requests the ticket ID for their issue or communication with support, we categorize the ticket under the "Ticket ID" tag
    Permitted/Prohibited Items: If a learner inquires about Approved or Restricted Items for the test, we can categorize their tickets under the "Permitted/Prohibited Items" tag.

    # CHAT CONVERSATION # 
    {chat_history}
    """

    try:
        response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                    "role": "user",
                    "content": prompt  
                    }
                ]
            )
    
        content = response.choices[0].message.content
        print("content:", content)
        print('\n')
        json_summary = eval(content.strip("```json"))
        print(f"No: {count}")
        print("Issue Description", chat_history)
        print('\n')
        # print ("Predicted tag: ", json_summary.get("tag", ""))
        print("--------------------------------------------------------")
        return json_summary.get("tag", "")
    except Exception as e:
        print(f"Error processing text: {chat_history}\nError: {e}")
        return ""

def process_excel(input_file, output_file):
    df = pd.read_excel(input_file)

    if 'predcited_tag' not in df.columns:
        df['predcited_tag'] = ""

    count = 1
    for index, row in df.iterrows():
        chat_text = row['Updated Description']
        
        if pd.notnull(chat_text):
            summary_of_issue = generate_summary(chat_text, count)
            df.at[index, 'predcited_tag'] = summary_of_issue

        count = count + 1

    df.to_excel(output_file, index=False)

input_file = r"/home/ajeet/Downloads/first_100_chats.xlsx" 
output_file = r"/home/ajeet/Downloads/new_tag_results_first_100.xlsx"
process_excel(input_file, output_file)
