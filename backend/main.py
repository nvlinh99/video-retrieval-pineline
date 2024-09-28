import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from pathlib import Path
from utils.helper import get_image_url, map_frame_to_video

app = Flask(__name__, template_folder='templates')
CORS(app)

def initialize_resources():
    global text_embedding, image_ebedding, transcript, description, translate_vi_to_en, get_surrounding_frames

    from utils.embedding.calc_text_embedding import text_embedding
    from utils.embedding.calc_image_embedding import image_ebedding
    from utils.transcript.transcript import transcript
    from utils.description.description import description
    from utils.helper import translate_vi_to_en
    from utils.search.search import get_surrounding_frames
    
@app.before_request
def setup():
    start_time = time.time()
    initialize_resources()
    end_time = time.time()
    print(f"Resource initialization took {end_time - start_time:.2f} seconds")

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the AIC HCMC 2024!"})

@app.route('/api/retrieval', methods=['POST'], strict_slashes=False)
def retrieval():
    data = request.get_json()
    text_input = translate_vi_to_en(data.get("input", ""))
    text_input = text_input.replace("vi ", "")

    method = data.get("method", "query")
    top_k = data.get("top_k", 10)
    result = None
    object_input = data.get('object_input', None)

    if method == "q&a":
        result = text_embedding(text_input, top_k, object_input)
    if method == "query":
        result = text_embedding(text_input, top_k, object_input)
    elif method == "image":
        text_input = os.path.join(os.path.dirname(__file__), data.get("input", ""))
        image_paths = [text_input]
        result = image_ebedding(image_paths, top_k, object_input)
    elif method == "transcript":
        result = transcript(text_input)
    elif method == "description":
        result = description(text_input, top_k, object_input)
    else:
        return jsonify({"message": "Invalid method"})
    
    data_respt = []
    matches = result.get('matches', None)
    if matches is not None:
        # file_path = 'output.txt'
        # # Open the file in write mode ('w')
        # with open(file_path, 'w') as file:
        #     print(matches, file=file)
        for match in matches:
            metadata = match.get('metadata', None)
            if metadata is not None:
                if method == "description":
                    #image_url = metadata.get('image_path', "").replace("keyframes", ".")
                    image_url = "./" + "/".join(metadata.get('image_path', "").split("/")[-4:])
                    frame_id = int(image_url.split("/")[-1].split(".")[0].split("_")[-1])
                    video_name = image_url.split("/")[2]
                    #print("image_path", metadata.get('image_path', "") , "image_url", get_image_url(image_url), "frame_id", frame_id, "video_name", video_name)
                    video_url = map_frame_to_video(video_name, frame_id)
                    data_respt.append(
                        {
                            "frame_name":  match.get('id', ""), 
                            "image_url": get_image_url(image_url), 
                            "frame_id": frame_id, 
                            "video_name": video_name, 
                            "score": match['score'],
                            "video_url": video_url
                        }
                    )
                else:
                    video_url = map_frame_to_video(metadata.get('video_name', ''), int(metadata.get('frame_name', 0)))
                    data_respt.append(
                        {
                            "frame_name":  match.get('id', ""), 
                            "image_url": get_image_url(match.get('id', "")), 
                            "frame_id": int(metadata.get('frame_name', 0)), 
                            "video_name": metadata.get('video_name', ''), 
                            "video_url": video_url
                        }
                    )
            else:
                video_url = map_frame_to_video(match.get('video_name', ''), int(match.get('frame_name', 0)))
                data_respt.append(
                    {
                        "frame_name":  match.get('id', ""),
                        "frame_id":  int(match.get('frame_name', 0)), 
                        "image_url": get_image_url(match.get('id', "")), 
                        "video_name": match.get('video_name', ""),
                        "video_url": video_url
                    }
                )
    else:
        data_respt.append(
            {
                "frame_name":  match.get('id', ""),
                "frame_id":  int(match.get('frame_name', 0)), 
                "image_url": get_image_url(match.get('id', "")), 
                "video_name": match.get('video_name', ""),
                "video_url": video_url
            }
        )

    return jsonify({"data": data_respt})

@app.route('/api/search', methods=['POST'], strict_slashes=False)
def search():
    data = request.get_json()
    mode = data.get("mode", "")
    frame_id = data.get("frame_id", "")
    frame_name = data.get("frame_name", "")
    video_name = data.get("video_name", "")
    data_respt = []

    if mode == "next":
        resp = get_surrounding_frames(frame_id, frame_name, video_name, window=50)
    else:
        resp = get_surrounding_frames(frame_id, frame_name, video_name, window=50)

    matches = resp['matches']
    if matches is not None:
        for match in matches:
            metadata = match['metadata']
            if metadata is not None:
                data_respt.append(
                        {
                            "frame_name":  match.get('id', ""), 
                            "image_url": get_image_url(match.get('id', "")), 
                            "frame_id": int(metadata.get('frame_name', 0)), 
                            "video_name": metadata.get('video_name', ''), 
                            "score": match['score']
                        }
                    )
            else:
                data_respt.append(
                    {
                        "frame_name":  match.get('frame_name', ""),
                        "frame_id":  match.get('id', ""), 
                        "image_url": get_image_url(match.get('id', "")), 
                        "video_name": match.get('video_name', "")
                    }
                )

    return jsonify({"data": data_respt})


# Running app
if __name__ == '__main__':
    app.run(port=8080)