import { Frame } from "@/apis/schema/model";
import { GetSearchResultResponse } from "@/apis/schema/search";
import { InputTypeEnum } from "@/constants";
import { Dispatch, SetStateAction } from "react";

export interface IObject {
  id: number;
  name: string;
  color: string;
  location: number;
  count: number;
}
export interface MainContext {
  inputValue: string;
  setInputValue: React.Dispatch<React.SetStateAction<string>>;
  activeInputType: InputTypeEnum;
  setActiveInputType: React.Dispatch<React.SetStateAction<InputTypeEnum>>;
  activeQueryIndex: number;
  setActiveQueryIndex: React.Dispatch<React.SetStateAction<number>>;
  queryDataList: IQueryData[];
  setQueryDataList: React.Dispatch<React.SetStateAction<IQueryData[]>>;
  onChangeInputValue: (value: string, type?: InputTypeEnum) => void;
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
  onSearch: () => void;
  onSubmit: () => void;
  onExport: () => void;
  isQA: boolean;
  setIsQA: React.Dispatch<React.SetStateAction<boolean>>;
  searchResultData?: GetSearchResultResponse;
  activePage: number;
  setActivePage: Dispatch<SetStateAction<number>>;
  isLoadingSearchResult: boolean;
  onDeleteFrame: (id: number) => void;
  onMoveFrameToTop: (frame: Frame) => void;
  onChangeFrameAnswer: (id: number, answer: string) => void;
  objects: IObject[];
  setObjects: React.Dispatch<React.SetStateAction<IObject[]>>;
}

export interface IQueryData {
  queryType: InputTypeEnum;
  queryInput: string;
  deletedFrames: number[];
  selectedFrames: Frame[];
  answers: Record<string, string>;
  isQA: boolean;
  objects: IObject[];
}
