import { GetFrameRelatedImagesQuery } from "@/apis/query/search";
import { Frame } from "@/apis/schema/model";
const generateImageURL = (index: number) =>
  `https://picsum.photos/${300 + index}/${300 + index}`;
export function generateFrames(count: number): Frame[] {
  // Helper function to generate image URL and ID

  // Generate all frames first
  const frames: Frame[] = Array.from({ length: count }, (_, index) => ({
    image_url: generateImageURL(index),
    video_url:
      "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    video_name: `video-${index}`,
    frame_id: index,
    score: 1,
    frame_name: `frame-name-${index}`,
  }));

  return frames;
}

export function generateFrameRelatedImages(query: GetFrameRelatedImagesQuery) {
  console.log("generateFrameRelatedImages", query);

  return generateFrames(7);
}
