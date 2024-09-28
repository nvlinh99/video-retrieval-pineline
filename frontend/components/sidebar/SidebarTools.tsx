import { useMainContext } from "@/contexts/main";
import {
  Box,
  Button,
  FormControlLabel,
  Stack,
  Switch,
  Typography,
} from "@mui/material";
import React from "react";

const SidebarTools = () => {
  const {
    queryDataList,
    activeQueryIndex,
    onRedo,
    onUndo,
    onReset,
    isQA,
    setIsQA,
  } = useMainContext();

  const canUndo = activeQueryIndex > 0;
  const canRedo = activeQueryIndex < queryDataList.length - 1;
  return (
    <Stack
      direction="row"
      sx={{
        alignItems: "center",
        width: "100%",
        justifyContent: "space-between",
      }}
      spacing={0.5}
    >
      <Box>
        <FormControlLabel
          sx={(theme) => ({ ...theme.typography.body2, mr: 0 })}
          control={
            <Switch checked={isQA} onChange={() => setIsQA((prev) => !prev)} />
          }
          disableTypography
          label="Q&A"
        />
      </Box>
      <Stack
        direction="row"
        spacing={0.5}
        sx={{
          alignItems: "center",
        }}
      >
        <Stack
          direction="row"
          spacing={1}
          sx={{
            alignItems: "center",
          }}
        >
          <Button
            variant="outlined"
            onClick={onUndo}
            size="small"
            disabled={!canUndo}
          >
            Undo
          </Button>
          <Typography
            sx={{
              flexShrink: 0,
            }}
            variant="body2"
          >
            {activeQueryIndex + 1} / {queryDataList.length}
          </Typography>
          <Button
            size="small"
            variant="outlined"
            onClick={onRedo}
            disabled={!canRedo}
          >
            Redo
          </Button>
        </Stack>
        <Button size="small" variant="outlined" onClick={onReset}>
          Reset
        </Button>
      </Stack>
    </Stack>
  );
};

export default SidebarTools;
