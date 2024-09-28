import { GetSearchResultResponse } from "@/apis/schema/search";
import { IQueryData } from "./types";
import { GetSearchResultQuery } from "@/apis/query/search";
import { capitalize } from "lodash";

export function getFinalFrames({
  activePage = 1,
  activeQuery,
  searchResultData,
  deletedFrameIds,
}: {
  searchResultData?: GetSearchResultResponse;
  activePage?: number;
  activeQuery: IQueryData;
  deletedFrameIds?: number[];
}) {
  const data = searchResultData?.data || [];
  if (!data.length || !activeQuery?.selectedFrames?.length) {
    return (
      data?.map((frame) => {
        return {
          ...frame,
          answer: activeQuery.answers[frame.frame_id],
        };
      }) || []
    );
  }
  const selectedFrameIds = activeQuery.selectedFrames.map(
    (frame) => frame.frame_id
  );
  return [
    ...(activePage > 1 ? [] : activeQuery.selectedFrames),
    ...data.filter(
      (frame) =>
        !selectedFrameIds.includes(frame.frame_id) &&
        !deletedFrameIds?.includes(frame.frame_id)
    ),
  ].map((frame) => {
    return {
      ...frame,
      answer: activeQuery.answers[frame.frame_id],
    };
  });
}

export function getSearchResultQuery({
  queryDataList,
  activeQueryIndex,
}: {
  queryDataList: IQueryData[];
  activeQueryIndex: number;
}): GetSearchResultQuery {
  const activeData = queryDataList[activeQueryIndex];

  if (!activeData) {
    return {};
  }
  const objects = activeData.objects || [];
  return {
    top_k: 200,
    input: activeData.queryInput,
    method: activeData.queryType,
    is_qa: activeData.isQA,
    object_name: objects.map((object) => capitalize(object.name)),
    object_color: objects.map((object) => object.color || "transparent"),
    object_location: objects.map((object) => (object.location || 1).toString()),
    object_count: objects.map((object) => (object.count || 1).toString()),
  };
}
