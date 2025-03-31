import os
import pandas as pd
import openai
import logging
from langchain_openai import OpenAI, OpenAIEmbeddings
import pandas as pd
from langchain.vectorstores import FAISS
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = ""
base_embeddings = OpenAIEmbeddings(openai_api_key=api_key)

client = OpenAI(api_key=api_key)

def clean_text(text):
    cleaned_text = " ".join(text.split())

    cleaned_text = (
        cleaned_text
        .replace(" :", ":")  # Remove spaces before colons
        .replace("https: //", "https://")
        .replace("http: //", "http://")
    )
    return cleaned_text

def retrieve_relevant_chunks(query, institution):
    """Retrieve relevant chunks using FAISS indexes."""
    results = []
    temp_dir = "/home/ajeet/codework/proctorgpt/faiss_indexes1/"
    institution_dir = os.path.join(temp_dir, f"{institution}faq")
    generic_dir = os.path.join(temp_dir, "genericfaq")

    # Try institution-specific index
    if os.path.exists(os.path.join(institution_dir, "index.faiss")):
        try:
            logger.info(f"Loading FAISS index for institution: {institution}")
            institution_index = FAISS.load_local(
                institution_dir, base_embeddings, allow_dangerous_deserialization=True
            )
            institution_results = institution_index.similarity_search(query, k=5)
            results.extend(institution_results)
        except Exception as e:
            logger.error(f"Error loading institution-specific FAISS index: {e}")

    # Use generic index if needed
    if len(results) < 5 and os.path.exists(os.path.join(generic_dir, "index.faiss")):
        try:
            remaining_k = 5 - len(results)
            logger.info("Loading generic FAISS index to retrieve additional results.")
            generic_index = FAISS.load_local(
                generic_dir, base_embeddings, allow_dangerous_deserialization=True
            )
            generic_results = generic_index.similarity_search(query, k=remaining_k)
            results.extend(generic_results)
        except Exception as e:
            logger.error(f"Error loading generic FAISS index: {e}")

    # Handle no results case
    if not results:
        logger.warning("No relevant information found.")
        return ""

    combined_context = "\n".join(result.page_content for result in results)
    return clean_text(combined_context)


# Step 2: Check if clarification is needed
clarification_prompt = """Given the 'user's question 'userQuestion' and the retrieved information in 'chunks', determine whether a clarifying question is required.

    # userQuestion # 
    {userQuestion}

    # chunks # 
    {chunks}

    # Output format #  
    Decision: '~' or '#'.

    INSTRUCTIONS:
    1. Analyse the 'chunks' and the 'user's question' to identify the specificity of the query and the scope of the information in 'chunks'.
    2. If the query is broad and the 'chunks' have multiple categories or types, output '#' and guide the chatbot to ask for clarification.
    3. If the query aligns well with a specific part of the 'chunks' that provides a comprehensive answer, output '~' 
"""

# Step 3: Generate the clarifying question if needed
clarifying_question_prompt = """Given the user question in 'userQuestion' and the information in 'chunks', construct a clarifying question that naturally incorporates these details to guide the user's response.

    # userQuestion # 
    {userQuestion}

    # CHUNKS # 
    {chunks}

    IMPORTANT: 
    Output the question and nothing else.
"""

def determine_clarification(question, context):
    """Determine whether a clarifying question is needed."""
    if not context:
        return "# (No context found, clarification needed.)"

    clarification_query = clarification_prompt.format(userQuestion=question, chunks=context)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant determining whether a clarifying question is needed."},
                {"role": "user", "content": clarification_query}
            ],
            temperature=0.7
        )

        clean = response = response.choices[0].message.content
        print(clean)
        return clean
    except Exception as e:
        logger.error(f"Error in OpenAI API call for clarification check: {e}")
        return "# (Error in API call, clarification needed.)"

def generate_clarifying_question(clarifying_check, question, context):
    """Generate a clarifying question based on the classification."""
    clarifying_query = clarifying_question_prompt.format(
        clarifyingQuestionCheck=clarifying_check, userQuestion=question, chunks=context
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant generating a clarifying question."},
                {"role": "user", "content": clarifying_query}
            ],
            temperature=0.7
        )

        clean = response = response.choices[0].message.content
        print(clean)
        return clean
    except Exception as e:
        logger.error(f"Error in OpenAI API call for clarifying question: {e}")
        return "Error generating clarifying question."

def process_question(question):
    """Pipeline: Retrieve chunks, classify if clarification is needed, then generate a clarifying question if required."""
    context = retrieve_relevant_chunks(question, "humber")

    clarifying_check = determine_clarification(question, context)

    if "#" in clarifying_check:
        clarifying_question = generate_clarifying_question(clarifying_check, question, context)
        return clarifying_question
    else:
        return "No clarification needed."
    

def process_xlsx_file(input_file, output_file):
    """Reads an Excel file, processes each query, and saves results with clarifying questions."""
    df = pd.read_excel(input_file
    df["clarify_question_3"] = df["Query"].apply(process_question)
    df.to_excel(output_file, index=False)

    print(f"Processed file saved as: {output_file}")


input_xlsx = "/home/ajeet/Downloads/Copy of GPT data set.xlsx"
output_xlsx = "/home/ajeet/Downloads/Copy of GPT data set.xlsx"

process_xlsx_file(input_xlsx, output_xlsx)


# question = "exam reschedule"
# result = process_question(question)
# print(result)
