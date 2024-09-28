from pyvi.ViTokenizer import tokenize
import torch
from sentence_transformers import SentenceTransformer
import re
import pandas as pd

from utils.helper import index_pinecone_image_description

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('dangvantuan/vietnamese-embedding')
model = model.to(device)

def remove_special_characters(desciption):
    desciption = desciption[0:800]
    desciption = desciption.replace(":00","h")
    desciption = re.sub(r'[^\w\s\,]', '', desciption)
    return desciption

def generate_embeddings(description):
    remove_special_characters(description)
    tokenized_desc = tokenize(description)

    splited_tokens = tokenized_desc.split(" ")
    splited_tokens = [tok for tok in splited_tokens if len(tok) < 25]

    if len(splited_tokens) > 200:
        tokenized_desc = " ".join(splited_tokens[0:200])
        print(f"Truncated description: {tokenized_desc}")
    else:
        tokenized_desc = " ".join(splited_tokens)

    embedding = model.encode(tokenized_desc, device=device).tolist()

    return embedding, tokenized_desc

def description(text, top_k=5, object_input=None):
    text_embedding, _ = generate_embeddings(text)
    index = index_pinecone_image_description()
    query_results = index.query(
        vector=text_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=True
    )
    return query_results

if __name__ == '__main__':
    desciption = "Generating embeddings"
    embedding, tokenized_desc = generate_embeddings(desciption)