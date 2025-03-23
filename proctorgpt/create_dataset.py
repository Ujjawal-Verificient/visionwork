import os
import pandas as pd
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document as Document
from langchain.schema import Document as Documentlangchain
from sentence_transformers import SentenceTransformer, util

OPENAI_API_KEY = "sk-"
client = OpenAI(api_key=OPENAI_API_KEY)


model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Example Questions from Dataset
example_tickets = pd.read_excel("/home/ajeet/Downloads/gpt_dataset.xlsx")
example_texts = example_tickets[['Query']].to_dict(orient='records')
example_descriptions = [ex['Query'] for ex in example_texts]
example_embeddings = model.encode(example_descriptions, convert_to_tensor=True)


def select_top_2_examples(chat_text):
    chat_embedding = model.encode(chat_text, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(chat_embedding, example_embeddings)[0]
    top_2_indices = similarities.argsort(descending=True)[:2]
    return [example_texts[i] for i in top_2_indices]

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

def qa_generator_llm(context, client):
    top_2_examples = select_top_2_examples(context)

    prompt = f"""
    Your task is to write a question and an answer given a context. Generate only one question. 
    The question should be phrased as if a user is asking informally—meaning it can be **incomplete question, vague, or not grammatically perfect**, yet still answerable from the context.  
Additionally, **do not always start the question with "sh-" words** (who, what, where, etc.); make it sound like a real user inquiry that may be indirect or implied.  

    ### Reference Questions:
    Example 1: {top_2_examples[0]}
    Example 2: {top_2_examples[1]}
    
    ### Format:
    Users Question: (A help-seeking question that requires some inference)  
    Answer: (A response derived from the context)  

    ### Context: {context}

    ### Output:::
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a Q&A generator."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-4",
        temperature=0.5,
        max_tokens=100
    )

    return chat_completion.choices[0].message.content

def generate_qa_dataset(doc_path, output_csv):
    file_ext = os.path.splitext(doc_path)[-1].lower()
    
    if file_ext == ".docx":
        text = extract_text_from_docx(doc_path)
    elif file_ext == ".pdf":
        text = extract_text_from_pdf(doc_path)
    else:
        print("Unsupported file format.")
        return

    chunks = text_splitter.split_text(text)

    qa_data = []
    count = 1
    for chunk in chunks:
        print(chunk)
        print("CHUNK")
        print(count)
        count = count + 1
        qa_text = qa_generator_llm(chunk, client)
        try:
            lines = qa_text.split("\n")
            question = answer = ""
            print(qa_text)

            for line in lines:
                if line.startswith("Users Question:"):
                    question = line.replace("Users Question:", "").strip()
                if  line.startswith("User's Question:"):
                    question = line.replace("User's Question:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()

            qa_data.append((question, chunk, answer))
        except Exception as e:
            print(f"Error processing chunk: {e}")

        print("-"*40)
        # break

    df = pd.DataFrame(qa_data, columns=["Question", "Context", "Answer"])
    df.to_csv(output_csv, index=False)
    print(f"Generated Q&A dataset saved as '{output_csv}'.")

generate_qa_dataset("/home/ajeet/Downloads/Copy of humberfaq.docx", "/home/ajeet/codework/proctorgpt/ajeesing/evaluation/qa_dataset_vague.csv")