import os
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from utils.helper import index_pinecone_transcript, index_pinecone_embedding
from utils.embedding.calc_text_embedding import get_sentencepiece_model_for_beit3, calc_text_embedding
model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

def embed_script(input_texts):
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy().tolist()
    return embeddings

def transcript_old(text, top_k=5, object_input=None):
    text_embedding = embed_script(text)
    index = index_pinecone_transcript()
    if object_input is not None:
        filter = {
            "object_labels": { "$in": object_input["object_name"] },
            "object_count": { "$in": object_input["object_count"] },
            "object_colors": { "$in": object_input["object_color"] },
            "ojbect_locations": { "$in": object_input["ojbect_location"] }
        }
        query_results = index.query(
            vector=text_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=True,
            filter=filter,
        )
        return query_results
    query_results = index.query(
        vector=text_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=True
    )
    return query_results


# Function to query video segment based on transcript
def find_video_segment_by_transcript(transcript: str):
    index = index_pinecone_transcript()    
    transcript_embedding= embed_script(transcript)[0]

    query_results = index.query(
        vector=transcript_embedding,
        top_k=5,  
        include_metadata=True,
        include_values=False
    )
    return query_results

# Function to query keyframe within a video segment based on metadata
def query_keyframe_in_video_segment(text_query, video_name, frame_start, frame_end):
    #tokenizer = get_sentencepiece_model_for_beit3(os.path.join(os.path.dirname(__file__), "beit3.spm"))
    text_embedding = calc_text_embedding(text_query)
    text_embedding = text_embedding[0].cpu().detach().numpy().tolist()

    # Apply metadata filter based on video_name (segment identifier)
    metadata_filter = {"video_name": {"$eq": video_name},
                       "$and":[{"frame_name" : {"$gte": frame_start}},{"frame_name" : {"$lte": frame_end}}]
                      }
    
    #print(metadata_filter)
    #metadata_filter = {'video_name': {'$eq': 'L02_V025'}, '$and': [{'frame_name': {'$gte': 7200.0}}, {'frame_name': {'$lte': 11700.0}}]}

    index = index_pinecone_embedding()
    # Query Pinecone for keyframes within this video segment
    # print(text_embedding)
    query_results = index.query(
        vector=text_embedding,  # Dummy vector for metadata query
        filter=metadata_filter,  # Filter based on video_name and other criteria
        top_k=50,  # Return top 10 most relevant keyframes
        include_metadata=True,
        include_values=False  # Include keyframe embeddings for further analysis
    )

    return query_results


def transcript(text):
    split_text = text.split("|")
    transcript_text = split_text[0]
    query_text = split_text[1]
    base_result = find_video_segment_by_transcript(transcript_text)
    result = []
    finalDict = dict()
    matches = base_result["matches"]
    if matches is not None and len(matches) > 0:
        for item in matches:
            video_name = item["metadata"]["video_name"]
            frame_start = item["metadata"]["frame_start"]
            frame_end = item["metadata"]["frame_end"]
            keyframe_result = query_keyframe_in_video_segment(query_text, video_name, frame_start, frame_end)
            result.append(keyframe_result)
    

    for item in result:
       i_matches = item["matches"]
       temp = []
       for i in i_matches:
            temp.append(i)
    
    finalDict["matches"] = temp

    return finalDict

if __name__ == '__main__':
    input_texts = "Hello, my name is John."
    embeddings = embed_script(input_texts)
    print(embeddings)