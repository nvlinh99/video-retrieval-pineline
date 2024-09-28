import queryString from "query-string";

export function getQueryString(
  queryParams: Record<string, any> = {},
  hasQuestionMark?: boolean
) {
  const value = queryString.stringify(queryParams);
  return hasQuestionMark && value ? `?${value}` : value;
}
