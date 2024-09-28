export enum InputTypeEnum {
  QA = "q&a",
  QUERY = "query",
  IMAGE = "image",
  TRANSCRIPT = "transcript",
  LOCALIZED = "localized",
  DESCRIPTION = "description",
}
export const inputTypes = [
  {
    label: "Q&A",
    value: InputTypeEnum.QA,
  },
  {
    label: "Query",
    value: InputTypeEnum.QUERY,
  },
  {
    label: "Image",
    value: InputTypeEnum.IMAGE,
  },
  {
    label: "TRANSCRIPT",
    value: InputTypeEnum.TRANSCRIPT,
  },
  // {
  //   label: "Localized",
  //   value: InputTypeEnum.LOCALIZED,
  // },
  {
    label: "Description",
    value: InputTypeEnum.DESCRIPTION,
  },
];

export enum QueryObjectOccurEnum {
  AND = "AND",
  OR = "OR",
}
export const queryObjectOccurs = [
  {
    label: "AND",
    value: QueryObjectOccurEnum.AND,
  },
  {
    label: "OR",
    value: QueryObjectOccurEnum.OR,
  },
];
export const DEFAULT_PAGE_SIZE = 100;
