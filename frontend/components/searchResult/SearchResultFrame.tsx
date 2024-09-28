import { Frame } from "@/apis/schema/model";
import { ArrowUpwardOutlined, CheckBoxOutlineBlank } from "@mui/icons-material";
import {
  Box,
  Button,
  Checkbox,
  Chip,
  FormControlLabel,
  Grid2,
  Skeleton,
  Stack,
} from "@mui/material";
import React from "react";
interface Props {
  item: Frame;
  onClick: (item: Frame) => void;
  isDeleted?: boolean;
  onMoveFrameToTop: (item: Frame) => void;
  onDeleteFrame: (id: number) => void;
}

function SearchResultFrame(props: Props) {
  const { item, onClick, onMoveFrameToTop, onDeleteFrame, isDeleted } = props;
  return (
    <Grid2
      size={{
        xs: 12,
        sm: 6,
        md: 4,
        lg: 3,
      }}
    >
      <Box
        sx={(theme) => ({
          position: "relative",
          width: "100%",
          paddingTop: "100%",
          ":hover .hover": {
            backgroundColor: theme.palette.action.hover,
          },
        })}
        onClick={() => {
          onClick(item);
        }}
      >
        <Box
          sx={{
            position: "absolute",
            width: "100%",
            height: "100%",
            objectFit: "cover",
            top: 0,
            left: 0,
          }}
          component={"img"}
          src={item.image_url}
          alt={item.frame_id.toString()}
        />
        <Box
          className="hover"
          sx={{
            position: "absolute",
            top: 0,
            left: 0,

            width: "100%",
            height: "100%",
            p: 1,
            backgroundColor: isDeleted
              ? `rgba(0,0,0,0.7) !important`
              : undefined,
            transition: "all 0.3s ease",
          }}
        >
          <Stack
            direction="row"
            spacing={1}
            sx={{
              justifyContent: "space-between",
            }}
          >
            <Chip label={item.frame_id} color="info" />
            <Stack
              sx={{
                alignItems: "flex-end",
              }}
            >
              <Button
                variant="contained"
                size="small"
                sx={{
                  minWidth: 0,
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onMoveFrameToTop(item);
                }}
              >
                <ArrowUpwardOutlined fontSize="small" />
              </Button>
              <FormControlLabel
                sx={(theme) => ({
                  ...theme.typography.caption,
                  color: "error.main",
                  mr: 0,
                })}
                control={
                  <Checkbox
                    color="error"
                    size="small"
                    checked={isDeleted || false}
                    sx={{
                      p: 1,
                    }}
                    onChange={(e) => {
                      e.stopPropagation();
                      onDeleteFrame(item.frame_id);
                    }}
                    icon={<CheckBoxOutlineBlank color="error" />}
                  />
                }
                label="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                }}
              />
            </Stack>
          </Stack>
        </Box>
      </Box>
    </Grid2>
  );
}
export function SearchResultFrameSkeleton() {
  return (
    <Grid2
      size={{
        xs: 12,
        sm: 6,
        md: 4,
        lg: 3,
      }}
    >
      <Skeleton
        variant="rectangular"
        sx={{
          width: "100%",
          paddingTop: "100%",
        }}
      />
    </Grid2>
  );
}

export default React.memo(SearchResultFrame);
