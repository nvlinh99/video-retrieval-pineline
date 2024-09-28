from utils.helper import index_pinecone_embedding

def get_surrounding_frames(frame_id, frame_name, video_name, window=50):
    index = index_pinecone_embedding()
    metadata_filter_01 = {
        "video_name": {"$eq": video_name},
        'frame_name': {"$gt": frame_name}
    }
    # lấy 50 frame sau
    query_results_01 = index.query(
        id=frame_id,
        filter=metadata_filter_01,
        top_k=window,
        include_metadata=True,
        include_values=False
    )
    
    metadata_filter_02 = {
        "video_name": {"$eq": video_name},
        'frame_name': {"$lte": frame_name}
    }

    query_results_02 = index.query(
        id=frame_id,
        filter=metadata_filter_02,
        top_k=window,
        include_metadata=True,
        include_values=False
    )
    
    final_result = [query_results_02['matches'],query_results_01['matches']]
    return final_result