import { Box, Typography } from "@mui/material";
import Image from "next/image";
import React from "react";

interface Props {
  color?: boolean;
  type: string;
  width?: number;
  height?: number;
}

const Icon = (props: Props) => {
  const { type, color, width, height } = props;
  return (
    <Box
      sx={{
        position: "relative",
        width: width || height,
        height: height || width,
      }}
    >
      <Image
        alt={type}
        src={`/images/icons/${type}.png`}
        fill={true}
        sizes="100%"
        className="rounded-md flex"
      />
      {color && (
        <Typography
          variant="caption"
          align="center"
          sx={{
            zIndex: 1,
            position: "absolute",
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
            fontSize: 10,
            textTransform: "capitalize",
            color: type === "black" ? "white" : undefined,
          }}
        >
          {type.replace("_", "")}
        </Typography>
      )}
    </Box>
  );
};

export default Icon;
