import { searchService } from "@/apis/search";
import { GetSearchResultQuery } from "@/apis/query/search";
import { useFetchWithCache } from "./useFetchWithCache";

export const KEY_INFINITE = "useFetchInfiniteSearchResults";

export const useFetchSearchResults = (query: GetSearchResultQuery) => {
  return useFetchWithCache(
    query.input || query.object_name?.length ? [KEY_INFINITE, query] : null,
    () => {
      return searchService.getResults({
        ...query,
      });
    }
  );
};
