import { Divider, Stack } from "@mui/material";
import React from "react";
import SidebarTools from "./SidebarTools";
import SidebarQueryTabs from "./SidebarQueryTabs";
import SidebarActions from "./SIdebarActtions";
import SidebarObjectInput from "./SidebarObjectInput";

const Sidebar = () => {
  return (
    <Stack
      sx={{
        width: "100%",
        height: "100%",
      }}
      spacing={1}
    >
      <SidebarTools />
      <Stack
        sx={{
          flex: 1,
          minHeight: 0,
        }}
      >
        <Stack
          sx={{
            height: "100%",
            overflow: "auto",
            overflowX: "hidden",
            scrollbarGutter: "stable",
          }}
          spacing={2}
        >
          <SidebarQueryTabs />
          <Divider flexItem />
          <SidebarObjectInput />
        </Stack>
      </Stack>
      <SidebarActions />
    </Stack>
  );
};

export default Sidebar;
