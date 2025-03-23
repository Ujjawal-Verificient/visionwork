import pandas as pd
import openai  # Ensure you have installed OpenAI's package
import time
from openai import OpenAI

OPENAI_API_KEY = "sk-proctorgpt-M2BT74NxByjHQA83wAMHT3BlbkFJJNBfmbw4pmoKVW6Doqzq"
client = OpenAI(api_key=OPENAI_API_KEY)

csv_file = "/home/ajeet/codework/proctorgpt/ajeesing/evaluation/qa_dataset_10rows.csv"  # Change this to your actual file name
df = pd.read_csv(csv_file)


EVALUATION_PROMPT = """### Task Description:
    You are given an instruction (which may include an input), a response to evaluate, a reference answer that gets a score of 5, and a score rubric for evaluation.

    1. Evaluate the response **strictly based on the score rubric**.
    2. Assign a score that is an **integer between 1 and 5**, referring to the score rubric.
    3. Output only the score as a single integer (no extra text or explanations).

    ### User's Question:
    {question}

    ### Response to evaluate:
    {response}

    ### Reference Answer (Score 5):
    {reference_answer}

    ### Score Rubric:
    **[Does the response align with the reference answer in terms of factual accuracy, completeness, and faithfulness?]**  
    - **Score 1:** Completely incorrect, inaccurate, incomplete, or contains hallucinated information.  
    - **Score 2:** Mostly incorrect, inaccurate, incomplete, or includes hallucinations.  
    - **Score 3:** Somewhat correct, accurate, and complete but with noticeable errors or missing key details.  
    - **Score 4:** Mostly correct, accurate, and complete, with minor issues or missing small details.  
    - **Score 5:** Completely correct, accurate, and faithfully aligned with the reference answer.  

    ### Output:
    """

# Function to get score from LLM
def get_score(question, response, reference_answer):
    prompt = EVALUATION_PROMPT.format(
        question=question,
        response=response,
        reference_answer=reference_answer
    )

    try:
        
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an evaluator tasked with assessing a response based on a given reference answer for a question. Your role is to strictly compare the response with the reference answer using the provided rubric and assign an accurate score between 1 and 5, outputting only the final score as an integer."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4",
            temperature=0.5,
            max_tokens=10 
        )
        score = completion.choices[0].message.content.strip()
        print(score)
        
        return int(score) if score.isdigit() and 1 <= int(score) <= 5 else None

    except Exception as e:
        print(f"Error: {e}")
        return None

scores = []
for _, row in df.iterrows():
    score = get_score(row["Question"], row["response"], row["Answer"])
    
    scores.append(score if score is not None else 0)

    time.sleep(1)

df["score"] = scores

df.to_csv(csv_file, index=False)
print(f"Scores saved")

average_score = sum(scores) / len(scores)
print(f"Average Score: {average_score:.2f}")