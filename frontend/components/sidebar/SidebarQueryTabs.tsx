import { useMainContext } from "@/contexts/main";
import React from "react";
import { Box, Button, Stack, TextField } from "@mui/material";
import { InputTypeEnum, inputTypes } from "@/constants";

const SidebarQueryTabs = () => {
  const {
    setActiveInputType,
    activeInputType,
    onChangeInputValue,
    inputValue,
    isQA,
  } = useMainContext();
  return (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Stack
          direction="row"
          sx={{
            flexWrap: "wrap",
            gap: 1,
          }}
        >
          {inputTypes.map((type) => {
            return (
              <Button
                size="small"
                key={type.value}
                onClick={() => setActiveInputType(type.value)}
                variant={
                  activeInputType === type.value ? "contained" : "outlined"
                }
              >
                {type.label}
              </Button>
            );
          })}
        </Stack>
      </Box>

      <Box
        sx={{
          p: 0,
          pt: 1,
        }}
      >
        <TextField
          multiline
          fullWidth
          value={inputValue || ""}
          onChange={(e) => onChangeInputValue(e.target.value, activeInputType)}
          rows={3}
          placeholder="Input your query here..."
          disabled={!isQA && activeInputType === InputTypeEnum.QA}
          slotProps={{
            htmlInput: {
              sx: {
                resize: "vertical",
              },
            },
          }}
        />
      </Box>
    </Box>
  );
};

export default SidebarQueryTabs;
