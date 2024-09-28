import { colors, icons } from "@/constants/icons";
import { useMainContext } from "@/contexts/main";
import { IObject } from "@/contexts/types";
import { Delete } from "@mui/icons-material";
import {
  Autocomplete,
  Box,
  Button,
  Chip,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { capitalize, range } from "lodash";
import React, { useState } from "react";

const objectNames = icons;
const objectColors = colors;
const objectLocations = range(1, 10).map((i) => i.toString());
const SidebarObjectInput = () => {
  const { setObjects, objects } = useMainContext();
  const [objectName, setObjectName] = useState<string>("");
  const [objectCount, setObjectCount] = useState(1);
  const [objectColor, setObjectColor] = useState<string>("");
  const [objectLocation, setObjectLocation] = useState<number>(1);
  const handleResetForm = () => {
    setObjectName("");
    setObjectCount(1);
    setObjectColor("");
    setObjectLocation(1);
  };
  const handleSave = () => {
    setObjects((prev) => {
      return [
        ...prev,
        {
          id: new Date().getTime(),
          name: objectName,
          color: objectColor,
          location: objectLocation,
          count: objectCount || 1,
        },
      ];
    });
    handleResetForm();
  };
  const handleDeleteObject = (object: IObject) => {
    setObjects((prev) => prev.filter((o) => o.id !== object.id));
  };

  const handleReset = () => {
    setObjects([]);
    handleResetForm();
  };
  return (
    <Stack spacing={2}>
      <Typography variant="body1">Objects</Typography>
      <Stack flexWrap="wrap" direction="row" gap={2} alignItems="center">
        {/* Object Name Autocomplete */}
        <Autocomplete
          size="small"
          options={objectNames}
          value={objectName}
          onChange={(_, newValue) => setObjectName(newValue)}
          renderInput={(params) => (
            <TextField {...params} label="Name" variant="outlined" />
          )}
          disableClearable
          sx={{
            width: 150,
          }}
        />

        {/* Object Color Autocomplete */}
        <Autocomplete
          size="small"
          options={objectColors}
          value={objectColor}
          onChange={(_, newValue) => setObjectColor(newValue || "")}
          renderInput={(params) => (
            <TextField {...params} label="Color" variant="outlined" />
          )}
          sx={{ minWidth: 120 }}
        />
        {/* Object Location Autocomplete */}
        <Autocomplete
          size="small"
          options={objectLocations}
          value={objectLocation?.toString()}
          onChange={(_, newValue) =>
            setObjectLocation(parseInt(newValue || "1"))
          }
          renderInput={(params) => (
            <TextField {...params} label="Location" variant="outlined" />
          )}
          sx={{ width: 90 }}
          disableClearable
        />

        {/* Object Count Input */}
        <TextField
          size="small"
          type="number"
          value={objectCount}
          onChange={(e) => setObjectCount(parseInt(e.target.value))}
          label="Count"
          variant="outlined"
          sx={{ width: 60 }}
        />

        {/* Save Button */}
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!objectName || !objectLocation}
        >
          Save
        </Button>
      </Stack>
      <Paper
        variant="outlined"
        sx={{
          p: 1.5,
          minHeight: 200,
          maxHeight: 400,
          overflow: "auto",
        }}
      >
        <Stack
          direction="row"
          sx={{
            flexWrap: "wrap",
            gap: 1,
          }}
        >
          {objects.map((object) => {
            return (
              <Chip
                size="small"
                key={object.id}
                label={[
                  capitalize(object.name),
                  object.color,
                  object.location,
                  object.count,
                ]
                  .filter(Boolean)
                  .join(", ")}
                onDelete={() => handleDeleteObject(object)}
              />
            );
          })}
        </Stack>
      </Paper>
      <Box>
        <Button
          color="error"
          startIcon={<Delete />}
          variant="contained"
          disabled={!objects.length}
          onClick={handleReset}
        >
          Reset
        </Button>
      </Box>
    </Stack>
  );
};

export default SidebarObjectInput;
