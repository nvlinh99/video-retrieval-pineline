import { PaginationQuery } from "./pagination";

export interface GetSearchResultQuery extends PaginationQuery {
  is_qa?: boolean;
  input?: string;
  method?: string;
  top_k?: number;
  object_name?: string[];
  object_color?: string[];
  object_location?: string[];
  object_count?: string[];
}

export interface GetFrameRelatedImagesQuery {
  video_name: string;
  frame_id: number;
  mode: "next" | "related";
  frame_name: string;
}
