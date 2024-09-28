"use client";

import SearchResult from "@/components/searchResult/SearchResult";
import Sidebar from "@/components/sidebar/Sidebar";
import { Box, Stack } from "@mui/material";

export default function Home() {
  return (
    <Box
      sx={{
        width: "100%",
        height: "100vh",
      }}
    >
      <Stack
        spacing={2}
        sx={{
          height: "100%",
        }}
        direction="row"
      >
        <Box
          sx={{
            borderRight: (theme) => `1px solid ${theme.palette.divider}`,
            height: "100%",
            p: 2,
            maxWidth: 420,
          }}
        >
          <Sidebar />
        </Box>
        <Box
          sx={{
            height: "100%",
            p: 2,
            flex: 1,
          }}
        >
          <SearchResult />
        </Box>
      </Stack>
    </Box>
  );
}
