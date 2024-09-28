import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import torch
import numpy as np
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle

# Load environment variables from .env file
load_dotenv()
PINECONE_EMBADDING_API_KEY = os.getenv('PINECONE_EMBADDING_API_KEY', '')
PINECONE_EMBADDING_ENVIRONMENT = os.getenv('PINECONE_EMBADDING_ENVIRONMENT', '')
PINECONE_EMBADDING_INDEX_NAME = os.getenv('PINECONE_EMBADDING_INDEX_NAME', '')


PINECONE_TRANSCRIPT_API_KEY = os.getenv('PINECONE_TRANSCRIPT_API_KEY', '')
PINECONE_TRANSCRIPT_ENVIRONMENT = os.getenv('PINECONE_TRANSCRIPT_ENVIRONMENT', '')
PINECONE_TRANSCRIPT_INDEX_NAME = os.getenv('PINECONE_TRANSCRIPT_INDEX_NAME', '')


PINECONE_DESCRIPTION_API_KEY = os.getenv('PINECONE_DESCRIPTION_API_KEY', '')
PINECONE_DESCRIPTION_ENVIRONMENT = os.getenv('PINECONE_DESCRIPTION_ENVIRONMENT', '')
PINECONE_DESCRIPTION_INDEX_NAME = os.getenv('PINECONE_DESCRIPTION_INDEX_NAME', '')

DOMAIN_S3 = os.getenv('DOMAIN_S3', '')

def index_pinecone_embedding():
    pc = Pinecone(api_key=PINECONE_EMBADDING_API_KEY)
    index = pc.Index(PINECONE_EMBADDING_INDEX_NAME)
    return index

def index_pinecone_transcript():
    pc = Pinecone(api_key=PINECONE_TRANSCRIPT_API_KEY)
    index = pc.Index(PINECONE_TRANSCRIPT_INDEX_NAME)
    return index

def index_pinecone_image_description():
    pc = Pinecone(api_key=PINECONE_DESCRIPTION_API_KEY)
    index = pc.Index(PINECONE_DESCRIPTION_INDEX_NAME)
    return index

def map_to_pinecone_format(first_segment, video_name):
    prefix = ""
    if first_segment in ["L01", "L02", "L03", "L04"]:
        prefix =  "keyframes/L01_L04/keyframes/" + video_name
    elif first_segment in ["L05", "L06", "L07", "L08"]:
        prefix =  "keyframes/L05_L08/keyframes/" + video_name
    elif first_segment in ["L09", "L10", "L11", "L12"]:
        prefix =  "keyframes/L09_L12/keyframes/" + video_name
    elif first_segment in ["L13", "L14", "L15", "L16"]:
        prefix =  "keyframes/L13_L16/keyframes/" + video_name
    elif first_segment in ["L17", "L18", "L19", "L20"]:
        prefix =  "keyframes/L17_L20/keyframes/" + video_name
    elif first_segment in ["L21", "L22", "L23", "L24"]:
        prefix =  "keyframes/L21_L24/keyframes/" + video_name
    else:
        prefix = prefix 
    return prefix


def get_image_url(image_path):
    path_without_dot = image_path.lstrip('./')
    segments = path_without_dot.split('/')
    first_segment = segments[0]
    return DOMAIN_S3 + map_to_pinecone_format(first_segment, path_without_dot)

def translate_vi_to_en(text):
    model_name = "VietAI/envit5-translation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Translate Using device: {device}")
    
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text



def map_to_pinecone_video_format(first_segment, video_name):
    prefix = ""
    if first_segment in ["L01", "L02", "L03", "L04"]:
        prefix =  "videos/L01_L04" + video_name
    elif first_segment in ["L05", "L06", "L07", "L08"]:
        prefix =  "videos/L05_L08" + video_name
    elif first_segment in ["L09", "L10", "L11", "L12"]:
        prefix =  "videos/L09_L12" + video_name
    elif first_segment in ["L13", "L14", "L15", "L16"]:
        prefix =  "videos/L13_L16" + video_name
    elif first_segment in ["L17", "L18", "L19", "L20"]:
        prefix =  "videos/L17_L20" + video_name
    elif first_segment in ["L21", "L22", "L23", "L24"]:
        prefix =  "videos/L21_L24" + video_name
    else:
        prefix = prefix 
    return prefix


pklFile = os.path.join(os.path.dirname(__file__), 'total_video_lst.pkl')

with open(pklFile, 'rb') as pkl_file:
    loaded_video_list = pickle.load(pkl_file)

def map_frame_to_video(video_name, frame_id):
    potential_videos = [video for video in loaded_video_list if video_name in video]
    for p_video in potential_videos:
        basename = os.path.basename(p_video)
        start = int(basename.split(".")[0].split("_")[2])
        end = int(basename.split(".")[0].split("_")[4])
        if(frame_id >= start) & (frame_id <= end):
            video_name = p_video.replace("/kaggle/working", "")
            first_segment = video_name.split("/")[1]
            return DOMAIN_S3 + map_to_pinecone_video_format(first_segment, video_name)
    return "none"
