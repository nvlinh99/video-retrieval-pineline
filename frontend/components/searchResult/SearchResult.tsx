import { useMainContext } from "@/contexts/main";
import { Box, Grid2, Pagination, Stack } from "@mui/material";
import React, { useCallback, useMemo, useState } from "react";
import { Repeated } from "../common/Repeated";
import SearchResultFrame, {
  SearchResultFrameSkeleton,
} from "./SearchResultFrame";
import { useDisclosure } from "@dwarvesf/react-hooks";
import { Frame } from "@/apis/schema/model";
import ViewFrameModal from "./ViewFrameModal";
import { getFinalFrames } from "@/contexts/mainUtils";
import { DEFAULT_PAGE_SIZE } from "@/constants";

const SearchResult = () => {
  const {
    searchResultData,
    activePage,
    setActivePage,
    isLoadingSearchResult,
    queryDataList,
    activeQueryIndex,
    onMoveFrameToTop,
    onDeleteFrame,
  } = useMainContext();
  const [activeFrame, setActiveFrame] = useState<Frame | null>(null);
  const viewDisclosure = useDisclosure();
  const activeQuery = queryDataList[activeQueryIndex];

  const { data, totalPage } = useMemo(() => {
    const allFrames = getFinalFrames({
      searchResultData,
      activePage,
      activeQuery,
    });
    return {
      data: allFrames.slice(
        (activePage - 1) * DEFAULT_PAGE_SIZE,
        activePage * DEFAULT_PAGE_SIZE
      ),
      totalPage: Math.ceil(allFrames.length / DEFAULT_PAGE_SIZE),
    };
  }, [searchResultData?.data, activePage, activeQuery?.selectedFrames]);
  const onClick = useCallback(
    (item: Frame) => {
      setActiveFrame(item);
      viewDisclosure.onOpen();
    },
    [viewDisclosure.onOpen]
  );
  return (
    <Stack
      sx={{
        height: "100%",
      }}
    >
      <Box
        sx={{
          flex: 1,
          overflow: "auto",
          minHeight: 0,
        }}
      >
        {!isLoadingSearchResult && !data.length && (
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              width: "100%",
              height: "100%",
            }}
          >
            Nothing here yet. Type something to begin your search.
          </Box>
        )}
        <Grid2 container spacing={2}>
          {isLoadingSearchResult && (
            <Repeated times={8} as={SearchResultFrameSkeleton} />
          )}

          {!isLoadingSearchResult &&
            !!data.length &&
            data.map((frame) => {
              return (
                <SearchResultFrame
                  key={frame.image_url}
                  item={frame}
                  onClick={onClick}
                  onMoveFrameToTop={onMoveFrameToTop}
                  onDeleteFrame={onDeleteFrame}
                  isDeleted={activeQuery?.deletedFrames.includes(
                    frame.frame_id
                  )}
                />
              );
            })}
        </Grid2>
      </Box>
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          mt: 2,
        }}
      >
        {totalPage ? (
          <Pagination
            count={totalPage}
            page={activePage}
            onChange={(_, page) => setActivePage(page)}
          />
        ) : null}
      </Box>
      {activeFrame && (
        <ViewFrameModal
          open={viewDisclosure.isOpen}
          onClose={viewDisclosure.onClose}
          frame={activeFrame}
        />
      )}
    </Stack>
  );
};

export default SearchResult;
