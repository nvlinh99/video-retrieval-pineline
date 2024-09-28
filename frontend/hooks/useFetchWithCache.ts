import useSWR, { Key, Fetcher, SWRConfiguration } from "swr";
import { useState, useEffect, useCallback } from "react";

export function useFetchWithCache<Data = any, Error = any>(
  key: Key,
  fn: Fetcher<Data> | null = null,
  config?: SWRConfiguration<Data, Error>
) {
  const { data, error, ...rest } = useSWR<Data, Error>(key, fn, config);
  const [internalData, setInternalData] = useState<Data | undefined>(data);
  const [loading, setLoading] = useState(!data && !error);

  const isFirstLoading = !internalData && !error;
  const clearInternalData = useCallback(() => {
    setInternalData(undefined);
  }, []);

  useEffect(() => {
    setLoading(!data && !error);
  }, [data, error]);

  useEffect(() => {
    if (data) {
      setInternalData(data);
    }
  }, [data]);

  return {
    data: key ? internalData : undefined,
    isFirstLoading,
    loading, // true if data is not available or not fetching
    error,
    ...rest,
    isLoading: !!key && loading, // true only when fetching data
    clearInternalData,
  };
}
