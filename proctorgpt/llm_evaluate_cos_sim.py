import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util


csv_file = "/home/ajeet/codework/proctorgpt/ajeesing/evaluation/qa_dataset_10rows.csv"
df = pd.read_csv(csv_file)

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity_score(response, reference_answer):
    response_embedding = model.encode(response, convert_to_tensor=True)
    reference_embedding = model.encode(reference_answer, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(response_embedding, reference_embedding).item()
    
    score = similarity
    
    return score 


scores = []
for _, row in df.iterrows():
    if pd.notna(row["response"]) and pd.notna(row["Answer"]): 
        score = compute_similarity_score(row["response"], row["Answer"])
    else:
        score = 0
    scores.append(score)


df["cosine_score"] = scores
df.to_csv(csv_file, index=False)

average_score = sum(scores) / len(scores)
print(f"Scores saved to {csv_file}")
print(f"Average Score: {average_score:.2f}")