import React, { useRef } from "react";
import { Box, Paper, Skeleton } from "@mui/material";
import Slider, { Settings } from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { Frame } from "@/apis/schema/model";

const settings: Settings = {
  dots: false,
  infinite: false,
  speed: 500,
  slidesToShow: 5, // Number of slides to show at a time
  slidesToScroll: 1, // Number of slides to scroll at once
  responsive: [
    {
      breakpoint: 600, // Adjust for mobile devices
      settings: {
        slidesToShow: 3,
        slidesToScroll: 1,
      },
    },
  ],
};

interface FrameRelatedImagesCarouselProps {
  frameRelatedImages?: Frame[]; // List of image URLs
  isLoading?: boolean;
  onImageClick?: (frame: Frame) => void;
  selectedFrame?: Frame | null;
}

const FrameRelatedImagesCarousel: React.FC<FrameRelatedImagesCarouselProps> = ({
  frameRelatedImages,
  isLoading,
  onImageClick,
  selectedFrame,
}) => {
  const isDraggingRef = useRef(false);

  if (!isLoading && !frameRelatedImages) {
    return null;
  }
  return (
    <Box
      sx={{
        width: "100%",
        flex: 1,
        minHeight: 0,
      }}
    >
      <Box sx={{ width: "100%", height: "100%", px: 2 }}>
        <Box
          component={Slider}
          sx={{
            height: "100%",
            ".slick-track, .slick-list": {
              height: "100%",
            },
            ".slick-slide > div": {
              height: "100%",
            },
          }}
          {...settings}
        >
          {isLoading &&
            Array.from({ length: 10 }).map((_, index) => (
              <Box
                key={index}
                sx={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  borderRadius: 4,
                  padding: 1,
                }}
              >
                <Skeleton
                  variant="rectangular"
                  width="100%"
                  height="100%"
                  sx={{
                    borderRadius: 4,
                    backgroundColor: "background.paper",
                  }}
                  animation="wave"
                />
              </Box>
            ))}
          {frameRelatedImages?.map((frame, index) => (
            <Box
              key={frame.image_url}
              sx={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                borderRadius: 4,

                pr: index < frameRelatedImages.length - 1 ? 2 : 0,
              }}
              onClick={(e) => {
                if (!isDraggingRef.current) {
                  e.preventDefault();

                  onImageClick?.(frame);
                }
              }}
              onMouseDown={() => {
                isDraggingRef.current = false;
              }}
              onMouseMove={() => {
                isDraggingRef.current = true;
              }}
            >
              <Paper
                elevation={10}
                sx={{
                  border: (theme) =>
                    selectedFrame?.image_url === frame.image_url
                      ? "4px solid " + theme.palette.success.main
                      : undefined,
                  height: "100%",
                  borderRadius: "6px",
                  backgroundColor: "transparent",
                  overflow: "hidden",
                  width: "100%",
                }}
              >
                <img
                  src={frame.image_url}
                  alt={`Related frame ${index + 1}`}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
              </Paper>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export default FrameRelatedImagesCarousel;
