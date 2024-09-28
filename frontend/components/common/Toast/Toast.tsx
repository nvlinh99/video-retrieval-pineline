"use client";

import React, {
  useState,
  ReactNode,
  useMemo,
  useCallback,
  useEffect,
  SyntheticEvent,
} from "react";
import Snackbar, {
  SnackbarCloseReason,
  SnackbarProps,
} from "@mui/material/Snackbar";
import Alert, { AlertProps } from "@mui/material/Alert";
import { createContext } from "@dwarvesf/react-utils";

const DEFAULT_DURATION = 3000; // 3 seconds

interface ToastContextValue {
  success: (msg?: string, config?: SnackbarProps) => void;
  error: (msg?: string, config?: SnackbarProps) => void;
  warning: (msg?: string, config?: SnackbarProps) => void;
  info: (msg?: string, config?: SnackbarProps) => void;
}

const [Provider] = createContext<ToastContextValue>({
  name: "ToastContext",
});

const toast: ToastContextValue = {
  success: () => {},
  error: () => {},
  warning: () => {},
  info: () => {},
};

const ToastProvider = ({ children }: { children: ReactNode }) => {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [severity, setSeverity] = useState<AlertProps["severity"]>("info");
  const [snackbarConfig, setSnackbarConfig] = useState<SnackbarProps>();

  const handleClose = (
    _event: Event | SyntheticEvent<any, Event>,
    reason: SnackbarCloseReason
  ) => {
    if (reason === "clickaway") {
      return;
    }

    setOpen(false);
  };

  const showToast = useCallback(
    (
      msg: string,
      type: AlertProps["severity"] = "info",
      config?: SnackbarProps
    ) => {
      setOpen(false);
      setTimeout(() => {
        setMessage(msg);
        setSeverity(type);
        setSnackbarConfig(config);
        setOpen(true);
      });
    },
    []
  );

  const success = useCallback(
    (msg?: string, config?: SnackbarProps) => {
      showToast(msg || "", "success", config);
    },
    [showToast]
  );
  const error = useCallback(
    (msg?: string, config?: SnackbarProps) => {
      showToast(msg || "", "error", config);
    },
    [showToast]
  );
  const warning = useCallback(
    (msg?: string, config?: SnackbarProps) => {
      showToast(msg || "", "warning", config);
    },
    [showToast]
  );
  const info = useCallback(
    (msg?: string, config?: SnackbarProps) => {
      showToast(msg || "", "info", config);
    },
    [showToast]
  );

  const contextValue = useMemo(
    () => ({ success, error, warning, info }),
    [success, error, warning, info]
  );

  useEffect(() => {
    Object.assign(toast, contextValue);
  }, [contextValue]);

  return (
    <Provider value={contextValue}>
      {children}
      <Snackbar
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
        autoHideDuration={DEFAULT_DURATION}
        open={open}
        onClose={handleClose}
        {...snackbarConfig}
      >
        <Alert
          elevation={6}
          severity={severity}
          sx={{
            color: "white",
          }}
          variant="filled"
          onClose={() => setOpen(false)}
        >
          {message}
        </Alert>
      </Snackbar>
    </Provider>
  );
};

export { toast, ToastProvider };
