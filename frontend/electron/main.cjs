const { app, BrowserWindow, ipcMain, session } = require("electron");
const path = require("path");
const { spawn, execFileSync } = require("child_process");
const http = require("http");
const fs = require("fs");
const { DEV_SERVER_URL } = require("./dev_runtime_config.cjs");
const { focusMainWindow } = require("./electron_window_lifecycle.cjs");
const {
  healthBelongsToManagedProcess,
  parseBackendHealth,
  reconcileBackendStatus,
  shouldStopBackendOnQuit,
} = require("./backend_supervisor_state.cjs");

const hasSingleInstanceLock = app.requestSingleInstanceLock();

let backendProc = null;
let backendOwnedByElectron = false;
let mainWindow = null;
let backendStartTime = 0;
let backendLastRestartTime = null;
let backendLastError = "";
let backendHistoricalError = "";
let backendStatus = "stopped";
let backendHealth = null;
let backendHealthCheckedAt = null;
let restartInFlight = null;
let backendConfig = null;
let healthReconcileTimer = null;

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
  return {
    devEnabled: isDevMode,
    ...reconcileBackendStatus({
      health: backendHealth,
      managedAlive: isBackendProcAlive(),
      managedPid: backendProc?.pid,
      managedStartTime: backendStartTime,
      status: backendStatus,
      lastRestartTime: backendLastRestartTime,
      lastError: backendLastError,
      historicalError: backendHistoricalError,
    }),
    healthCheckedAt: backendHealthCheckedAt,
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
      const health = parseBackendHealth(await httpRequest(url));
      if (health) return health;
    } catch (_) {}
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return null;
}

async function readBackendHealth() {
  try {
    const response = await httpRequest(getBackendConfig().healthUrl);
    let health = parseBackendHealth(response);
    if (!health) {
      health = parseBackendHealth(response, getBackendListenerIdentity());
    }
    return health;
  } catch (_) {
    return null;
  }
}

function getBackendListenerIdentity() {
  if (process.platform !== "win32") return null;
  const script = [
    "$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1",
    "if ($c) {",
    "  $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$c.OwningProcess)",
    "  $started = if ($p -and $p.CreationDate) { [DateTimeOffset]::new($p.CreationDate).ToUnixTimeMilliseconds() } else { 0 }",
    "  [pscustomobject]@{ pid=[int]$c.OwningProcess; parent_pid=[int]$p.ParentProcessId; uptime_ms=[Math]::Max(0, [DateTimeOffset]::Now.ToUnixTimeMilliseconds() - $started) } | ConvertTo-Json -Compress",
    "}",
  ].join("; ");
  try {
    const output = execFileSync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-Command", script],
      { encoding: "utf8", windowsHide: true, timeout: 2500 },
    ).trim();
    return output ? JSON.parse(output) : null;
  } catch (_) {
    return null;
  }
}

async function refreshBackendStatus({ emit = true } = {}) {
  const health = await readBackendHealth();
  backendHealth = health;
  backendHealthCheckedAt = new Date().toISOString();
  if (health) {
    backendStatus = "healthy";
    backendLastError = "";
  } else if (!isBackendProcAlive() && !["restarting", "stopping"].includes(backendStatus)) {
    backendStatus = backendStatus === "failed" ? "failed" : "stopped";
  }
  return emit ? emitDevStatus() : getBackendStatus();
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

  const proc = spawn(py, args, {
    cwd: backendDir,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env,
  });

  backendProc = proc;
  backendOwnedByElectron = true;
  backendHealth = null;
  backendStartTime = Date.now();
  backendLastRestartTime = new Date().toISOString();
  devLog(`backend launcher spawned pid=${proc.pid}`);
  emitDevStatus();

  proc.on("error", (error) => {
    const message = String(error?.message || error);
    backendHistoricalError = message;
    if (backendProc === proc) {
      backendLastError = message;
      backendStatus = "failed";
    }
    console.error("[backend] spawn error:", error);
    emitDevStatus();
  });

  proc.on("exit", (code, signal) => {
    const message = code && code !== 0 ? `Backend exited with code ${code}` : "";
    if (message) backendHistoricalError = message;
    const isCurrent = backendProc === proc;
    if (isCurrent) {
      backendProc = null;
      backendOwnedByElectron = false;
      backendStartTime = 0;
    }
    void refreshBackendStatus({ emit: false }).then(() => {
      if (!backendHealth && isCurrent && (!backendProc || backendProc === proc)) {
        backendLastError = message;
        backendStatus = message ? "failed" : "stopped";
      }
      emitDevStatus();
    });
  });

  proc.stdout.on("data", (d) => console.log("[backend]", d.toString().trim()));
  proc.stderr.on("data", (d) => console.log("[backend:err]", d.toString().trim()));
  return proc;
}

async function requestBackendShutdown() {
  try {
    await httpRequest(getBackendConfig().shutdownUrl, { method: "POST", timeoutMs: 2000 });
  } catch (_) {}
}

function killProcessTree(candidatePid, force = true) {
  const pid = Number(candidatePid);
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) return false;
  try {
    const args = ["/PID", String(pid), "/T"];
    if (force) args.push("/F");
    execFileSync("taskkill", args, { stdio: "ignore" });
    return true;
  } catch (_) {
    if (backendProc?.pid === pid) {
      try { backendProc.kill(); } catch (_) {}
    }
    return false;
  }
}

function killBackendTree(force = true) {
  const actualPid = backendHealth?.pid;
  const launcherPid = backendProc?.pid;
  let killed = false;
  if (actualPid) killed = killProcessTree(actualPid, force) || killed;
  if (launcherPid && launcherPid !== actualPid) killed = killProcessTree(launcherPid, force) || killed;
  return killed;
}

async function stopBackendGracefully(timeoutMs = 7000) {
  const health = await readBackendHealth();
  backendHealth = health;
  const proc = backendProc;
  if (!health && (!proc || !proc.pid)) {
    backendStatus = "stopped";
    backendLastError = "";
    emitDevStatus();
    return true;
  }

  const actualPid = health?.pid || null;
  const launcherPid = proc?.pid || null;
  devLog(`stopping backend actual_pid=${actualPid || "none"} launcher_pid=${launcherPid || "none"}`);
  backendStatus = "stopping";
  emitDevStatus({ status: "stopping" });

  await requestBackendShutdown();
  if (actualPid) killProcessTree(actualPid, true);
  if (launcherPid && launcherPid !== actualPid) killProcessTree(launcherPid, true);

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!(await readBackendHealth())) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  if (backendProc === proc) {
    backendProc = null;
  }
  backendOwnedByElectron = false;
  backendHealth = null;
  backendStartTime = 0;
  backendStatus = "stopped";
  backendLastError = "";
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

      const health = await waitForBackend(getBackendConfig().healthUrl, 120, 250);
      if (!health) throw new Error("Backend health check timed out");
      if (!isBackendProcAlive()) {
        throw new Error("Backend process exited during startup. Port 8000 may already be in use.");
      }
      if (!healthBelongsToManagedProcess(health, proc.pid)) {
        throw new Error(`Backend health belongs to unexpected process ${health.pid}`);
      }

      backendHealth = health;
      backendHealthCheckedAt = new Date().toISOString();
      backendStatus = "healthy";
      backendLastError = "";
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
  ipcMain.handle("dev:get-backend-status", () => refreshBackendStatus());

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
  let health = await readBackendHealth();
  if (health) {
    backendOwnedByElectron = false;
    backendHealth = health;
    backendStatus = "healthy";
    backendLastError = "";
    devLog(`adopted existing backend actual_pid=${health.pid} parent_pid=${health.parent_pid || "none"}`);
  } else {
    const proc = startBackend();
    health = proc ? await waitForBackend(getBackendConfig().healthUrl, 360, 250) : null;
    if (health && healthBelongsToManagedProcess(health, proc.pid)) {
      backendHealth = health;
      backendStatus = "healthy";
      backendLastError = "";
      devLog(`health_check ok actual_pid=${health.pid} launcher_pid=${proc.pid}`);
    } else {
      backendStatus = "failed";
      backendLastError = health
        ? `Backend health belongs to unexpected process ${health.pid}`
        : "Backend did not become healthy on startup";
    }
  }
  backendHealthCheckedAt = new Date().toISOString();
  emitDevStatus();
  console.log("[backend] health:", backendHealth ? "OK" : "NOT READY");

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

  win.webContents.once("did-finish-load", () => {
    if (isDevMode) devLog(`renderer_url=${win.webContents.getURL()}`);
  });

  healthReconcileTimer = setInterval(() => {
    void refreshBackendStatus();
  }, 2000);

  if (isDevMode) {
    win.loadURL(DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    focusMainWindow(mainWindow);
  });

  app.whenReady().then(() => {
    installDevIpc();
    return createWindow();
  });

  app.on("before-quit", () => {
    if (healthReconcileTimer) clearInterval(healthReconcileTimer);
    if (shouldStopBackendOnQuit({ ownedByElectron: backendOwnedByElectron })) {
      killBackendTree(true);
    }
  });

  app.on("window-all-closed", () => {
    app.quit();
  });
}
