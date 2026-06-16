const { app, BrowserWindow, ipcMain, session } = require("electron");
const path = require("path");
const { spawn, execFileSync } = require("child_process");
const http = require("http");
const fs = require("fs");

let backendProc = null;
let mainWindow = null;
let backendStartTime = 0;
let backendLastRestartTime = null;
let backendLastError = "";
let backendStatus = "stopped";
let restartInFlight = null;
let backendConfig = null;

const isDevMode = process.env.ELECTRON_DEV === "1" || process.env.HEBE_DEV_CONTROLS === "1";

function devLog(message) {
  console.log(`[HEBE][DEV] ${message}`);
}

function backendLog(message) {
  console.log(`[backend] ${message}`);
}

function getBackendConfig() {
  if (backendConfig) return backendConfig;
  const backendDir = path.resolve(__dirname, "../../backend");
  const py = path.join(backendDir, ".venv", "Scripts", "python.exe");
  backendConfig = {
    backendDir,
    py,
    args: ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    healthUrl: "http://127.0.0.1:8000/health",
    shutdownUrl: "http://127.0.0.1:8000/dev/shutdown",
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
  };
  if (process.env.HEBE_BACKEND_UVICORN_RELOAD === "1") {
    backendConfig.args.push("--reload");
  }
  return backendConfig;
}

function isBackendProcAlive() {
  return Boolean(
    backendProc &&
    backendProc.pid &&
    backendProc.exitCode === null &&
    backendProc.signalCode === null
  );
}

function getBackendStatus(extra = {}) {
  const now = Date.now();
  return {
    devEnabled: isDevMode,
    running: isBackendProcAlive(),
    pid: isBackendProcAlive() ? backendProc.pid : null,
    uptimeMs: isBackendProcAlive() && backendStartTime ? now - backendStartTime : 0,
    lastRestartTime: backendLastRestartTime,
    lastError: backendLastError,
    status: backendStatus,
    ...extra,
  };
}

function emitDevStatus(extra = {}) {
  const payload = getBackendStatus(extra);
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("dev:backend-status", payload);
  }
  return payload;
}

function allowMediaPermissions() {
  const ses = session.defaultSession;

  ses.setPermissionRequestHandler((wc, permission, cb) => {
    if (permission === "media") return cb(true);
    cb(false);
  });

  ses.setPermissionCheckHandler((wc, permission) => {
    if (permission === "media") return true;
    return false;
  });
}

function httpRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      url,
      { method: options.method || "GET", timeout: options.timeoutMs || 2500 },
      (res) => {
        let body = "";
        res.on("data", (chunk) => { body += chunk.toString(); });
        res.on("end", () => resolve({ statusCode: res.statusCode || 0, body }));
      },
    );
    req.on("timeout", () => req.destroy(new Error("request timeout")));
    req.on("error", reject);
    if (options.body) req.write(options.body);
    req.end();
  });
}

async function waitForBackend(url, tries = 80, delayMs = 150) {
  for (let i = 0; i < tries; i++) {
    try {
      const { statusCode } = await httpRequest(url);
      if (statusCode >= 200 && statusCode < 500) return true;
    } catch (_) {}
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return false;
}

function startBackend() {
  const { backendDir, py, args, env } = getBackendConfig();

  if (isBackendProcAlive()) {
    backendLog(`already running pid=${backendProc.pid}`);
    return backendProc;
  }

  backendLog(`backendDir: ${backendDir}`);
  backendLog(`python: ${py} exists: ${fs.existsSync(py)}`);

  if (!fs.existsSync(py)) {
    backendLastError = "Backend venv python not found. Create backend/.venv first.";
    backendStatus = "failed";
    console.error("[backend]", backendLastError);
    emitDevStatus();
    return null;
  }

  backendStatus = "starting";
  emitDevStatus();

  backendProc = spawn(py, args, {
    cwd: backendDir,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env,
  });

  backendStartTime = Date.now();
  backendLastRestartTime = new Date().toISOString();
  devLog(`backend spawned pid=${backendProc.pid}`);
  emitDevStatus();

  backendProc.on("error", (error) => {
    backendLastError = String(error?.message || error);
    backendStatus = "failed";
    console.error("[backend] spawn error:", error);
    emitDevStatus();
  });

  backendProc.on("exit", (code, signal) => {
    if (backendProc) {
      backendStatus = code === 0 || signal ? "stopped" : "failed";
      if (code && code !== 0) backendLastError = `Backend exited with code ${code}`;
      emitDevStatus();
    }
  });

  backendProc.stdout.on("data", (d) => console.log("[backend]", d.toString().trim()));
  backendProc.stderr.on("data", (d) => console.log("[backend:err]", d.toString().trim()));
  return backendProc;
}

async function requestBackendShutdown() {
  try {
    await httpRequest(getBackendConfig().shutdownUrl, { method: "POST", timeoutMs: 2000 });
  } catch (_) {}
}

function killBackendTree(force = true) {
  if (!backendProc || !backendProc.pid) return false;
  const pid = backendProc.pid;
  try {
    const args = ["/PID", String(pid), "/T"];
    if (force) args.push("/F");
    execFileSync("taskkill", args, { stdio: "ignore" });
    return true;
  } catch (_) {
    try { backendProc.kill(); } catch (_) {}
    return false;
  }
}

async function stopBackendGracefully(timeoutMs = 7000) {
  if (!backendProc || !backendProc.pid) {
    backendStatus = "stopped";
    emitDevStatus();
    return true;
  }

  const proc = backendProc;
  const pid = proc.pid;
  devLog(`stopping backend pid=${pid}`);
  backendStatus = "stopped";
  emitDevStatus({ status: "stopping" });

  await requestBackendShutdown();
  try { proc.kill("SIGTERM"); } catch (_) {}

  const stopped = await new Promise((resolve) => {
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      resolve(ok);
    };
    proc.once("exit", () => finish(true));
    setTimeout(() => finish(false), timeoutMs);
  });

  if (!stopped) {
    killBackendTree(true);
  }

  if (backendProc === proc) {
    backendProc = null;
  }
  backendStartTime = 0;
  backendStatus = "stopped";
  devLog("backend stopped");
  emitDevStatus();
  return true;
}

async function restartBackend() {
  if (!isDevMode) {
    return { ok: false, status: "failed", error: "Dev controls are disabled." };
  }
  if (restartInFlight) return restartInFlight;

  restartInFlight = (async () => {
    devLog("restart_backend requested");
    backendLastError = "";
    backendStatus = "restarting";
    emitDevStatus();

    try {
      await stopBackendGracefully();
      backendStatus = "starting";
      emitDevStatus();

      const proc = startBackend();
      if (!proc) throw new Error(backendLastError || "Backend spawn failed");

      const healthy = await waitForBackend(getBackendConfig().healthUrl, 120, 250);
      if (!healthy) throw new Error("Backend health check timed out");
      if (!isBackendProcAlive()) {
        throw new Error("Backend process exited during startup. Port 8000 may already be in use.");
      }

      backendStatus = "healthy";
      devLog("health_check ok");
      emitDevStatus();
      return getBackendStatus({ ok: true });
    } catch (error) {
      backendLastError = String(error?.message || error);
      backendStatus = "failed";
      emitDevStatus();
      return getBackendStatus({ ok: false, error: backendLastError });
    } finally {
      restartInFlight = null;
    }
  })();

  return restartInFlight;
}

function installDevIpc() {
  ipcMain.handle("dev:get-backend-status", () => getBackendStatus());

  ipcMain.handle("dev:reload-ui", (event) => {
    if (!isDevMode) return { ok: false, error: "Dev controls are disabled." };
    event.sender.reloadIgnoringCache();
    return { ok: true };
  });

  ipcMain.handle("dev:restart-backend", async () => restartBackend());

  ipcMain.handle("dev:full-reset", async (event) => {
    if (!isDevMode) return { ok: false, error: "Dev controls are disabled." };
    const result = await restartBackend();
    if (result.ok) {
      event.sender.reloadIgnoringCache();
      devLog("full_reset complete");
    }
    return result;
  });
}

async function createWindow() {
  allowMediaPermissions();
  startBackend();

  const ok = await waitForBackend(getBackendConfig().healthUrl, 360, 250);
  if (ok) {
    backendStatus = "healthy";
    devLog("health_check ok");
  } else {
    backendStatus = "failed";
    backendLastError = "Backend did not become healthy on startup";
  }
  emitDevStatus();
  console.log("[backend] health:", ok ? "OK" : "NOT READY");

  const win = new BrowserWindow({
    width: 1200,
    height: 760,
    backgroundColor: "#070916",
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });
  mainWindow = win;

  if (isDevMode) {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

app.whenReady().then(() => {
  installDevIpc();
  createWindow();
});

app.on("before-quit", () => {
  killBackendTree(true);
});

app.on("window-all-closed", () => {
  killBackendTree(true);
  app.quit();
});
