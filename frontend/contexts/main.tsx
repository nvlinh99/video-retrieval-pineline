"use client";

import { DEFAULT_PAGE_SIZE, InputTypeEnum } from "@/constants";
import { createContext } from "@dwarvesf/react-utils";
import {
  PropsWithChildren,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { IObject, IQueryData, MainContext } from "./types";
import { getFinalFrames, getSearchResultQuery } from "./mainUtils";
import { toast } from "@/components/common/Toast";
import { useEventCallback } from "@/hooks/useEventCallback";
import useDebouncedCallback from "@/hooks/useDebouncedCallback";
import { useFetchSearchResults } from "@/hooks/useFetchSearchResults";
import { xor } from "lodash";
import { Frame } from "@/apis/schema/model";
import exportFramesCSV from "@/utils/csv";
import { removeFileExtension } from "@/utils/file";

const [Provider, useMainContext] = createContext<MainContext>({
  name: "main",
});

export const MainContextProvider = ({ children }: PropsWithChildren) => {
  const [isQA, setIsQA] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [activeInputType, setActiveInputType] = useState<InputTypeEnum>(
    InputTypeEnum.QA
  );
  const [activeQueryIndex, setActiveQueryIndex] = useState(-1);
  const [queryDataList, setQueryDataList] = useState<IQueryData[]>([]);
  const [objects, setObjects] = useState<IObject[]>([]);
  const [activePage, setActivePage] = useState(1);

  const query = useMemo(() => {
    return getSearchResultQuery({ queryDataList, activeQueryIndex });
  }, [queryDataList, activeQueryIndex]);

  const { data: searchResultData, isLoading: isLoadingSearchResult } =
    useFetchSearchResults(query);
  const onChangeInputValue = (value: string, _type = activeInputType) => {
    setInputValue(value);
  };

  const onUndo = () => {
    setActiveQueryIndex((prev) => Math.max(0, prev - 1));
  };
  const onRedo = () => {
    setActiveQueryIndex((prev) => Math.min(queryDataList.length - 1, prev + 1));
  };
  const onDeleteFrame = useCallback(
    (id: number) => {
      setQueryDataList((prev) => {
        const updated = [...prev];
        const lastItem = updated[activeQueryIndex];
        if (!lastItem) {
          return prev;
        }
        updated[activeQueryIndex] = {
          ...lastItem,
          deletedFrames: xor(lastItem?.deletedFrames, [id]),
        };
        return updated;
      });
      saveDataToStorage();
    },
    [activeQueryIndex]
  );

  const onMoveFrameToTop = useCallback(
    (frame: Frame) => {
      setQueryDataList((prev) => {
        const updated = [...prev];
        const lastItem = updated[activeQueryIndex];
        if (!lastItem) {
          return prev;
        }
        const newSelectedFrames = lastItem.selectedFrames.filter(
          (i) => i.frame_id !== frame.frame_id
        );
        updated[activeQueryIndex] = {
          ...lastItem,
          selectedFrames: [frame, ...newSelectedFrames],
        };
        return updated;
      });
      saveDataToStorage();
    },
    [activeQueryIndex]
  );
  const onChangeFrameAnswer = (id: number, answer: string) => {
    setQueryDataList((prev) => {
      const updated = [...prev];
      const lastItem = updated[activeQueryIndex];
      if (!lastItem) {
        return prev;
      }
      updated[activeQueryIndex] = {
        ...lastItem,
        answers: {
          ...(lastItem.answers || {}),
          [id]: answer,
        },
      };
      return updated;
    });
  };
  const onReset = () => {
    setQueryDataList([]);
    setActiveQueryIndex(-1);
    setObjects([]);
    setInputValue("");
    setActiveInputType(InputTypeEnum.QA);
    setIsQA(false);
    saveDataToStorage();
  };

  const onSearch = () => {
    setQueryDataList((prev) => {
      const updated = [...prev];
      updated.push({
        queryType: activeInputType,
        queryInput: inputValue,
        deletedFrames: [],
        selectedFrames: [],
        answers: {},
        isQA,
        objects,
      });
      return updated;
    });
    setActiveQueryIndex((prev) => prev + 1);
    saveDataToStorage();
  };
  const onSubmit = () => {
    toast.info("Coming soon!");
  };
  const onExport = async () => {
    try {
      exportFramesCSV(
        getFinalFrames({
          searchResultData: searchResultData,
          activeQuery: queryDataList[activeQueryIndex],
          activePage,
          deletedFrameIds: queryDataList[activeQueryIndex]?.deletedFrames,
        }).slice(0, DEFAULT_PAGE_SIZE),

        (frame) => {
          const result = `${removeFileExtension(frame.video_name)},${
            frame.frame_id
          }`;
          if (isQA && frame.answer) {
            return `${result},${frame.answer}`;
          }
          return result;
        }
      );
    } catch (error: any) {
      console.log(error);
      toast.error(error.message);
    }
  };

  const syncStateWithActiveQuery = (data: IQueryData[], index: number) => {
    const activeData = data[index];
    if (!activeData) {
      return;
    }
    setInputValue(
      typeof activeData.queryInput === "string" ? activeData.queryInput : ""
    );
    setActiveInputType(activeData.queryType);
    setObjects(activeData.objects || []);
    setIsQA(activeData.isQA || false);
  };
  const saveDataToStorage = useDebouncedCallback(() => {
    localStorage.setItem("queryDataList", JSON.stringify(queryDataList));
    localStorage.setItem("activeQueryIndex", activeQueryIndex.toString());
  }, 1000);

  const syncDataFromStorage = useEventCallback(() => {
    const queryDataList = localStorage.getItem("queryDataList");
    const data = queryDataList ? JSON.parse(queryDataList) : [];
    if (queryDataList) {
      setQueryDataList(data);
    }
    const activeQueryIndex = localStorage.getItem("activeQueryIndex");
    const index = activeQueryIndex
      ? parseInt(activeQueryIndex)
      : (queryDataList?.length ?? 0) - 1;
    setActiveQueryIndex(index);
    if (queryDataList) {
      syncStateWithActiveQuery(data, index);
    }
  });

  useEffect(() => {
    syncDataFromStorage();
  }, [syncDataFromStorage]);

  const store = {
    inputValue,
    setInputValue,
    activeInputType,
    setActiveInputType,
    activeQueryIndex,
    setActiveQueryIndex,
    queryDataList,
    setQueryDataList,
    onChangeInputValue,
    onUndo,
    onRedo,
    onReset,
    isQA,
    setIsQA,
    onSearch,
    onSubmit,
    onExport,

    searchResultData,
    activePage,
    setActivePage,
    isLoadingSearchResult,
    onDeleteFrame,
    onMoveFrameToTop,
    onChangeFrameAnswer,
    objects,
    setObjects,
  };

  return <Provider value={store}>{children}</Provider>;
};

export { useMainContext };
