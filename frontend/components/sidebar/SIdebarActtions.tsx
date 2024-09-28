import { Button, Stack } from "@mui/material";
import React, { useState } from "react";

import { useMainContext } from "@/contexts/main";
import { LoadingButton } from "@mui/lab";

const SidebarActions = () => {
  const { onSearch, onSubmit, onExport } = useMainContext();
  const [isLoading, setIsLoading] = useState(false);
  return (
    <Stack
      direction="row"
      spacing={1}
      sx={{
        alignItems: "center",
        width: "100%",
        justifyContent: "space-between",
      }}
    >
      <LoadingButton
        loading={isLoading}
        variant="contained"
        onClick={async () => {
          setIsLoading(true);
          await onSearch();
          setIsLoading(false);
        }}
      >
        Search
      </LoadingButton>
      <LoadingButton
        loading={isLoading}
        variant="contained"
        onClick={async () => {
          setIsLoading(true);
          await onSubmit();
          setIsLoading(false);
        }}
      >
        Submit
      </LoadingButton>
      <LoadingButton
        loading={isLoading}
        variant="contained"
        onClick={async () => {
          setIsLoading(true);
          await onExport();
          setIsLoading(false);
        }}
      >
        Export
      </LoadingButton>
    </Stack>
  );
};

export default SidebarActions;
