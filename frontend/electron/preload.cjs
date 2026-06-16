const { contextBridge, ipcRenderer } = require("electron");

const devEnabled = process.env.ELECTRON_DEV === "1" || process.env.HEBE_DEV_CONTROLS === "1";

if (devEnabled) {
  contextBridge.exposeInMainWorld("hebeDev", {
    enabled: true,
    reloadUi: () => ipcRenderer.invoke("dev:reload-ui"),
    restartBackend: () => ipcRenderer.invoke("dev:restart-backend"),
    fullReset: () => ipcRenderer.invoke("dev:full-reset"),
    getBackendStatus: () => ipcRenderer.invoke("dev:get-backend-status"),
    onBackendStatus: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on("dev:backend-status", listener);
      return () => ipcRenderer.removeListener("dev:backend-status", listener);
    },
  });
} else {
  contextBridge.exposeInMainWorld("hebeDev", {
    enabled: false,
  });
}
