export interface Frame {
  frame_id: number;
  image_url: string;
  score: number;
  video_name: string;
  video_url: string;
  frame_name: string;

  // only form export
  answer?: string;
}

export interface Pagination {
  totalPage: number;
  page: number;
  pageSize: number;
}
