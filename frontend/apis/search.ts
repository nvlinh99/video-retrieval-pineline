import { Client } from "./client";
import {
  GetFrameRelatedImagesQuery,
  GetSearchResultQuery,
} from "./query/search";
import { generateFrameRelatedImages, generateFrames } from "@/utils/dummyData";
import {
  GetFrameRelatedImagesResponse,
  GetSearchResultResponse,
} from "./schema/search";
import fetcher from "./fetcher";
import { sleep } from "@dwarvesf/react-utils";

export const searchResultDummyData = generateFrames(1000);
class SearchService extends Client {
  public getResults(payload: GetSearchResultQuery) {
    if (!this.baseUrl) {
      return sleep(0).then(() => {
        return {
          data: searchResultDummyData,
        };
      });
    }

    return fetcher<GetSearchResultResponse>(`${this.baseUrl}/api/retrieval`, {
      headers: this.privateHeaders,
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
  public getFrameRelatedImages(payload: GetFrameRelatedImagesQuery) {
    if (!this.baseUrl) {
      return sleep(0).then(() => {
        return {
          data: generateFrameRelatedImages(payload),
        };
      });
    }

    return fetcher<GetFrameRelatedImagesResponse>(
      `${this.baseUrl}/api/retrieval`,
      {
        headers: this.privateHeaders,
        method: "POST",
        body: JSON.stringify(payload),
      }
    );
  }
}

const searchService = new SearchService();

export { searchService };
