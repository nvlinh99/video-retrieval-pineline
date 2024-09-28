"use client";

import { createTheme } from "@mui/material/styles";

export const MuiTheme = createTheme({
  components: {
    MuiButton: {
      defaultProps: {
        size: "small",
        sx: {
          ...(theme) => theme.typography.body2,
        },
      },
    },
  },
});
