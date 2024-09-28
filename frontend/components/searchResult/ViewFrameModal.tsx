import { Frame } from "@/apis/schema/model";
import { useMainContext } from "@/contexts/main";
import { Close } from "@mui/icons-material";
import {
  Box,
  Button,
  Dialog,
  DialogContent,
  FormControlLabel,
  Grid2,
  IconButton,
  Radio,
  RadioGroup,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import React, { useEffect, useState } from "react";
import { toast } from "../common/Toast";
import { useFetchFrameRelatedImages } from "@/hooks/useFetchFrameRelatedImages";
import FrameRelatedImagesCarousel from "./FrameRelatedImagesCarousel";
import { GetFrameRelatedImagesQuery } from "@/apis/query/search";
import { useEventCallback } from "@/hooks/useEventCallback";
import { sleep } from "@dwarvesf/react-utils";

interface Props {
  frame: Frame;
  open: boolean;
  onClose: () => void;
}

const ViewFrameModal = (props: Props) => {
  const { frame, open, onClose } = props;
  const [selectedFrame, setSelectedFrame] = useState<Frame | null>(null);
  const { isQA, queryDataList, activeQueryIndex, onChangeFrameAnswer } =
    useMainContext();
  const activeQuery = queryDataList[activeQueryIndex];
  const [internalAnswer, setInternalAnswer] = useState("");
  const [videoName, setVideoName] = useState("");
  const [frameId, setFrameId] = useState("");
  const [frameName, setFrameName] = useState("");
  const [mode, setMode] = useState<"image" | "video">("image");
  const [frameOption, setFrameOption] = useState<"next" | "related">("next");
  const [searchFrameQuery, setSearchFrameQuery] =
    useState<GetFrameRelatedImagesQuery | null>(null);
  const { data: { data: frameRelatedImages } = {}, isLoading } =
    useFetchFrameRelatedImages(searchFrameQuery);

  const handleSearch = useEventCallback(() => {
    setSearchFrameQuery({
      video_name: videoName,
      frame_id: parseInt(frameId),
      mode: frameOption,
      frame_name: frameName,
    });
  });

  useEffect(() => {
    if (frame) {
      setFrameId(frame.frame_id.toString());
      setFrameName(frame.frame_name || "");
      setVideoName(frame.video_name || "");
      setInternalAnswer(activeQuery?.answers?.[frame.frame_id] || "");
      setMode("image");
      setFrameOption("next");
      sleep(100).then(() => {
        handleSearch();
      });
    }
  }, [frame]);
  const activeFrame = selectedFrame || frame;
  return (
    <Dialog
      maxWidth={false}
      open={open}
      onClose={onClose}
      fullScreen
      PaperProps={{
        sx: {
          background: "transparent",
          boxShadow: "none",
          alignSelf: "flex-start",
          p: 1,
          pt: 3,
        },
      }}
    >
      <IconButton
        onClick={onClose}
        color="error"
        sx={{
          position: "absolute",
          right: 0,
          top: 0,
        }}
        size="small"
      >
        <Close
          sx={{
            width: 28,
            height: 28,
          }}
        />
      </IconButton>
      <DialogContent>
        <Stack
          spacing={2}
          sx={{
            height: "100%",
          }}
        >
          <Box
            sx={{
              height: "70%",
              minHeight: 500,
            }}
          >
            <Grid2
              container
              spacing={2}
              sx={{
                height: "100%",
              }}
            >
              {/* Left Panel */}
              <Grid2
                size={2.5}
                sx={{
                  height: "100%",
                }}
              >
                <Box
                  sx={{
                    backgroundColor: "background.paper",
                    padding: 2,
                    height: "100%",
                  }}
                >
                  {/* Mode view */}
                  <Typography variant="h6">Mode view</Typography>
                  <RadioGroup
                    row
                    value={mode}
                    onChange={(e) =>
                      setMode(e.target.value as "image" | "video")
                    }
                  >
                    <FormControlLabel
                      value="image"
                      control={<Radio />}
                      label="View image"
                    />
                    <FormControlLabel
                      value="video"
                      control={<Radio />}
                      label="View video"
                    />
                  </RadioGroup>

                  {/* View frame */}
                  <Typography variant="h6">View frame</Typography>
                  <RadioGroup
                    row
                    value={frameOption}
                    onChange={(e) =>
                      setFrameOption(e.target.value as "next" | "related")
                    }
                  >
                    <FormControlLabel
                      value="next"
                      control={<Radio />}
                      label="Show next frames"
                    />
                    <FormControlLabel
                      value="related"
                      control={<Radio />}
                      label="Show related frames"
                    />
                  </RadioGroup>

                  {/* Search option */}
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      View frame
                    </Typography>
                    <TextField
                      size="small"
                      fullWidth
                      label="Video name"
                      value={videoName}
                      onChange={(e) => setVideoName(e.target.value)}
                      sx={{ marginBottom: 2 }}
                    />
                    <TextField
                      size="small"
                      fullWidth
                      label="Frame ID"
                      value={frameId}
                      onChange={(e) => setFrameId(e.target.value)}
                      sx={{ marginBottom: 2 }}
                    />
                    <TextField
                      size="small"
                      fullWidth
                      label="Frame name"
                      value={frameName}
                      onChange={(e) => setFrameName(e.target.value)}
                      sx={{ marginBottom: 2 }}
                    />

                    <Button
                      fullWidth
                      variant="contained"
                      onClick={handleSearch}
                    >
                      Search
                    </Button>
                  </Box>
                </Box>
              </Grid2>

              {/* Main View */}
              <Grid2
                size={7}
                sx={{
                  height: "100%",
                }}
              >
                <Box
                  sx={{
                    backgroundColor: "background.paper",
                    height: "100%",
                  }}
                >
                  {/* Placeholder for video/image view */}
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      height: "100%",
                    }}
                  >
                    {mode === "video" && activeFrame.video_url ? (
                      <Box
                        sx={{
                          width: "100%",
                          height: "100%",
                          position: "relative",
                        }}
                      >
                        <Box
                          component={"video"}
                          controls
                          sx={{
                            width: "100%",
                            height: "100%",
                            objectFit: "cover",
                          }}
                        >
                          <source src={activeFrame.video_url} />
                          Your browser does not support the video tag.
                        </Box>
                      </Box>
                    ) : activeFrame.image_url ? (
                      <Box
                        component={"img"}
                        sx={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                        }}
                        src={activeFrame.image_url}
                        alt="Frame"
                      />
                    ) : (
                      <Typography variant="body1">
                        No frame available
                      </Typography>
                    )}
                  </Box>
                </Box>
              </Grid2>

              {/* Right Panel */}
              <Grid2
                size={2.5}
                sx={{
                  height: "100%",
                }}
              >
                <Box
                  sx={{
                    backgroundColor: "background.paper",
                    padding: 2,
                    height: "100%",
                  }}
                >
                  {/* Metadata */}
                  <Typography variant="h6" gutterBottom>
                    Metadata
                  </Typography>
                  <Box
                    sx={{
                      backgroundColor: "#fff",
                      minHeight: 150,
                      border: "1px solid #ccc",
                      marginBottom: 2,
                      p: 1,
                    }}
                  >
                    <MetaDataRow
                      label="Video name"
                      value={activeFrame.video_name}
                    />
                    <MetaDataRow
                      label="Frame ID"
                      value={activeFrame.frame_id}
                    />
                    <MetaDataRow
                      label="Frame name"
                      value={activeFrame.frame_name}
                    />
                    {isQA && (
                      <MetaDataRow
                        label="Answer"
                        value={activeQuery?.answers[activeFrame.frame_id]}
                      />
                    )}
                  </Box>

                  {isQA && (
                    <Box>
                      {/* Q&A Answer */}
                      <Typography variant="h6" gutterBottom>
                        Q&A Answer
                      </Typography>
                      <TextField
                        fullWidth
                        label="Answer"
                        multiline
                        rows={4}
                        sx={{ marginBottom: 2 }}
                        size="small"
                        placeholder="answer"
                        value={internalAnswer}
                        onChange={(e) => setInternalAnswer(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            onChangeFrameAnswer(frame.frame_id, internalAnswer);
                            toast.success("Answer updated");
                          }
                        }}
                      />
                      <Button
                        disabled={!internalAnswer}
                        fullWidth
                        variant="contained"
                        onClick={() => {
                          onChangeFrameAnswer(frame.frame_id, internalAnswer);
                          toast.success("Answer updated");
                        }}
                      >
                        Submit
                      </Button>
                    </Box>
                  )}
                </Box>
              </Grid2>
            </Grid2>
          </Box>
          <FrameRelatedImagesCarousel
            isLoading={isLoading}
            frameRelatedImages={frameRelatedImages}
            onImageClick={(selectedFrame) =>
              setSelectedFrame((prev) =>
                prev?.frame_id === selectedFrame.frame_id ? null : selectedFrame
              )
            }
            selectedFrame={selectedFrame}
          />
        </Stack>
      </DialogContent>
    </Dialog>
  );
};

export default ViewFrameModal;

function MetaDataRow(props: { label: string; value?: React.ReactNode }) {
  const { label, value } = props;

  return (
    <Box sx={{ display: "flex", flexDirection: "row", gap: 1 }}>
      <Typography
        variant="body2"
        sx={{
          flexShrink: 0,
        }}
      >
        {label}:{" "}
      </Typography>
      {value && (
        <Typography
          variant="body2"
          fontWeight={600}
          sx={{
            wordBreak: "break-all",
          }}
        >
          {value}
        </Typography>
      )}
    </Box>
  );
}
