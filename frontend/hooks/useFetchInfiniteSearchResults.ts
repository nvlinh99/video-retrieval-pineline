import { searchService } from "@/apis/search";
import { useFetchInfinite } from "./useFetchInfinite";
import { GetSearchResultQuery } from "@/apis/query/search";

export const KEY_INFINITE = "useFetchInfiniteSearchResults";

export const useFetchInfiniteSearchResults = (query: GetSearchResultQuery) => {
  return useFetchInfinite(
    (index) => {
      return [KEY_INFINITE, query, index + 1];
    },
    (key) => {
      const [page] = key?.slice(-1) || [];
      return searchService.getResults({
        ...query,
        page,
      });
    }
  );
};
