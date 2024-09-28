import { searchService } from "@/apis/search";
import { useFetchWithCache } from "./useFetchWithCache";
import { GetFrameRelatedImagesQuery } from "@/apis/query/search";

export const KEY_INFINITE = "useFetchInfiniteSearchResults";

export const useFetchFrameRelatedImages = (
  query: GetFrameRelatedImagesQuery | null
) => {
  return useFetchWithCache(
    query && query.video_name && query.frame_id && query.frame_name
      ? [KEY_INFINITE, query]
      : null,
    () => {
      return searchService.getFrameRelatedImages({
        ...query!,
      });
    }
  );
};
