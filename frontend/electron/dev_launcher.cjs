const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync, spawn } = require("child_process");

const {
  DEV_SERVER_HOST,
  DEV_SERVER_PORT,
  DEV_SERVER_URL,
} = require("./dev_runtime_config.cjs");
const {
  buildElectronSpawnOptions,
  canonicalPath,
  canTerminateVerifiedVite,
  decidePreflight,
} = require("./dev_launcher_core.cjs");

const workspace = path.resolve(__dirname, "..");
const workspaceKey = crypto.createHash("sha256").update(canonicalPath(workspace)).digest("hex").slice(0, 16);
const ownershipPath = path.join(os.tmpdir(), `hebe-dev-${workspaceKey}.json`);
const token = crypto.randomUUID();
let viteProc = null;
let electronProc = null;
let shuttingDown = false;

function log(message) {
  console.log(`[HEBE][DEV_LAUNCHER] ${message}`);
}

function processSnapshot() {
  if (process.platform !== "win32") {
    throw new Error("The Hebe DEV ownership preflight currently requires Windows.");
  }
  const script = `
$connection = Get-NetTCPConnection -LocalPort ${DEV_SERVER_PORT} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$all = Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('node.exe', 'electron.exe') }
function Convert-Process($item) {
  if (-not $item) { return $null }
  $started = if ($item.CreationDate) { [DateTimeOffset]::new($item.CreationDate).ToUnixTimeMilliseconds() } else { 0 }
  return [pscustomobject]@{
    pid = [int]$item.ProcessId
    parentPid = [int]$item.ParentProcessId
    name = [string]$item.Name
    executablePath = [string]$item.ExecutablePath
    commandLine = [string]$item.CommandLine
    startedAtMs = [long]$started
  }
}
$listener = $null
if ($connection) {
  $listener = Convert-Process (Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$connection.OwningProcess))
}
[pscustomobject]@{
  listener = $listener
  processes = @($all | ForEach-Object { Convert-Process $_ })
} | ConvertTo-Json -Compress -Depth 5
`;
  const output = execFileSync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    { encoding: "utf8", windowsHide: true, timeout: 5000 },
  ).trim();
  const parsed = JSON.parse(output || "{}");
  return {
    listener: parsed.listener || null,
    processes: Array.isArray(parsed.processes)
      ? parsed.processes
      : parsed.processes ? [parsed.processes] : [],
  };
}

function readOwnershipRecord() {
  try {
    const record = JSON.parse(fs.readFileSync(ownershipPath, "utf8"));
    return canonicalPath(record.workspace) === canonicalPath(workspace) ? record : null;
  } catch (_) {
    return null;
  }
}

function writeOwnershipRecord(record) {
  const temporary = `${ownershipPath}.${process.pid}.${token}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(record, null, 2), "utf8");
  fs.renameSync(temporary, ownershipPath);
}

function removeOwnershipRecord() {
  try {
    const record = readOwnershipRecord();
    if (record && record.token === token) fs.unlinkSync(ownershipPath);
  } catch (_) {}
}

function terminateProcessTree(pid) {
  const target = Number(pid);
  if (!Number.isInteger(target) || target <= 0 || target === process.pid) return false;
  try {
    execFileSync("taskkill.exe", ["/PID", String(target), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
      timeout: 5000,
    });
    return true;
  } catch (_) {
    return false;
  }
}

function focusExistingElectron(pid) {
  const target = Number(pid);
  if (!Number.isInteger(target) || target <= 0) return false;
  const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class HebeDevWindow {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr extraData);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowTextLength(IntPtr hWnd);
  public static bool Focus(uint targetPid) {
    IntPtr titled = IntPtr.Zero;
    IntPtr visible = IntPtr.Zero;
    IntPtr fallback = IntPtr.Zero;
    EnumWindows(delegate(IntPtr hWnd, IntPtr unused) {
      uint ownerPid;
      GetWindowThreadProcessId(hWnd, out ownerPid);
      if (ownerPid != targetPid) return true;
      if (fallback == IntPtr.Zero) fallback = hWnd;
      if (visible == IntPtr.Zero && IsWindowVisible(hWnd)) visible = hWnd;
      if (GetWindowTextLength(hWnd) > 0) {
        titled = hWnd;
        return false;
      }
      return true;
    }, IntPtr.Zero);
    IntPtr match = titled != IntPtr.Zero ? titled : (visible != IntPtr.Zero ? visible : fallback);
    if (match == IntPtr.Zero) return false;
    ShowWindow(match, 9);
    SetForegroundWindow(match);
    return IsWindowVisible(match) && !IsIconic(match);
  }
}
'@
if ([HebeDevWindow]::Focus(${target})) { exit 0 }
exit 2
`;
  try {
    execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
      stdio: "ignore",
      windowsHide: true,
      timeout: 5000,
    });
    return true;
  } catch (_) {
    return false;
  }
}

async function waitFor(predicate, { timeoutMs, intervalMs = 100 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await predicate();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return null;
}

async function waitForCanonicalVite() {
  return waitFor(async () => {
    try {
      const response = await fetch(DEV_SERVER_URL);
      const body = await response.text();
      return response.ok && body.includes("<title>Hebe UI</title>") && body.includes("/@vite/client");
    } catch (_) {
      return false;
    }
  }, { timeoutMs: 20000, intervalMs: 150 });
}

async function waitForPortFree(timeoutMs = 7000) {
  return waitFor(() => !processSnapshot().listener, { timeoutMs, intervalMs: 150 });
}

async function cleanupVerifiedStaleVite(expected) {
  const current = processSnapshot().listener;
  if (!canTerminateVerifiedVite({ expected, current, workspace })) {
    throw new Error("stale_vite_ownership_changed; refusing to terminate listener");
  }
  log(`cleanup stale_vite pid=${current.pid} started_at_ms=${current.startedAtMs}`);
  if (!terminateProcessTree(current.pid)) {
    throw new Error(`stale_vite_cleanup_failed pid=${current.pid}`);
  }
  if (!(await waitForPortFree())) {
    throw new Error(`stale_vite_port_release_timeout port=${DEV_SERVER_PORT}`);
  }
  try {
    const record = readOwnershipRecord();
    if (record && canonicalPath(record.workspace) === canonicalPath(workspace)) fs.unlinkSync(ownershipPath);
  } catch (_) {}
}

function spawnVite() {
  const viteEntry = path.join(workspace, "node_modules", "vite", "bin", "vite.js");
  const child = spawn(
    process.execPath,
    [
      viteEntry,
      "--host", DEV_SERVER_HOST,
      "--port", String(DEV_SERVER_PORT),
      "--strictPort",
      "--mode", `hebe-dev-${token}`,
    ],
    {
      cwd: workspace,
      env: {
        ...process.env,
        HEBE_DEV_INSTANCE_TOKEN: token,
        HEBE_DEV_SERVER_URL: DEV_SERVER_URL,
        HEBE_DEV_WORKSPACE: workspace,
      },
      stdio: "inherit",
      windowsHide: true,
    },
  );
  log(`vite_spawned pid=${child.pid} url=${DEV_SERVER_URL}`);
  return child;
}

function spawnElectron() {
  const executable = path.join(workspace, "node_modules", "electron", "dist", "electron.exe");
  const electronEnv = {
    ...process.env,
    ELECTRON_DEV: "1",
    HEBE_DEV_INSTANCE_TOKEN: token,
    HEBE_DEV_SERVER_URL: DEV_SERVER_URL,
    HEBE_DEV_WORKSPACE: workspace,
  };
  delete electronEnv.ELECTRON_RUN_AS_NODE;
  const child = spawn(
    executable,
    [workspace, `--hebe-dev-token=${token}`],
    buildElectronSpawnOptions({ workspace, env: electronEnv }),
  );
  log(`electron_spawned pid=${child.pid}`);
  return child;
}

async function recordOwnedProcesses() {
  const snapshot = processSnapshot();
  const vite = snapshot.processes.find((item) => Number(item.pid) === Number(viteProc?.pid));
  const electron = snapshot.processes.find((item) => Number(item.pid) === Number(electronProc?.pid));
  if (!vite || !electron) throw new Error("owned_process_identity_unavailable");
  writeOwnershipRecord({
    version: 1,
    token,
    workspace,
    launcher_pid: process.pid,
    vite_pid: vite.pid,
    vite_started_at_ms: vite.startedAtMs,
    electron_pid: electron.pid,
    electron_started_at_ms: electron.startedAtMs,
    dev_server_url: DEV_SERVER_URL,
    created_at: new Date().toISOString(),
  });
}

async function closeElectronGracefully() {
  if (!electronProc || electronProc.exitCode !== null) return;
  const pid = electronProc.pid;
  try {
    const script = `$p = Get-Process -Id ${Number(pid)} -ErrorAction Stop; if ($p.MainWindowHandle -ne 0) { [void]$p.CloseMainWindow() }`;
    execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
      stdio: "ignore", windowsHide: true, timeout: 3000,
    });
  } catch (_) {}
  const exited = await waitFor(() => electronProc.exitCode !== null, { timeoutMs: 4000, intervalMs: 100 });
  if (!exited) terminateProcessTree(pid);
}

async function cleanupOwnedFrontend(reason) {
  if (shuttingDown) return;
  shuttingDown = true;
  log(`shutdown reason=${reason}`);
  await closeElectronGracefully();
  if (viteProc && viteProc.exitCode === null) terminateProcessTree(viteProc.pid);
  removeOwnershipRecord();
}

async function run() {
  const initial = processSnapshot();
  const record = readOwnershipRecord();
  const decision = decidePreflight({ ...initial, record, workspace });
  log(`preflight action=${decision.action} reason=${decision.reason}`);
  if (decision.action === "fail") {
    const owner = decision.listener || {};
    throw new Error(
      `port_in_use_foreign_process port=${DEV_SERVER_PORT} pid=${owner.pid || "unknown"} `
      + `name=${owner.name || "unknown"} command=${JSON.stringify(owner.commandLine || "")}`,
    );
  }
  if (decision.action === "focus_existing") {
    const focused = focusExistingElectron(decision.electron.pid);
    log(`existing_instance pid=${decision.electron.pid} focus=${focused ? "ok" : "requested"}`);
    return;
  }
  if (decision.action === "cleanup_stale") {
    await cleanupVerifiedStaleVite(decision.listener);
  }

  viteProc = spawnVite();
  viteProc.once("exit", (code, signal) => {
    if (!shuttingDown) {
      console.error(`[HEBE][DEV_LAUNCHER] vite_exited code=${code} signal=${signal || "none"}`);
      void cleanupOwnedFrontend("vite_exit").finally(() => { process.exitCode = code || 1; });
    }
  });
  if (!(await waitForCanonicalVite())) {
    throw new Error(`vite_start_timeout url=${DEV_SERVER_URL}`);
  }
  const listener = processSnapshot().listener;
  if (!listener || Number(listener.pid) !== Number(viteProc.pid)) {
    throw new Error(`vite_listener_ownership_mismatch expected=${viteProc.pid} actual=${listener?.pid || "none"}`);
  }

  electronProc = spawnElectron();
  electronProc.once("exit", (code, signal) => {
    log(`electron_exited code=${code} signal=${signal || "none"}`);
    void cleanupOwnedFrontend("electron_exit").finally(() => { process.exitCode = code || 0; });
  });
  const electronReady = await waitFor(
    () => processSnapshot().processes.find((item) => Number(item.pid) === Number(electronProc.pid)),
    { timeoutMs: 10000, intervalMs: 150 },
  );
  if (!electronReady) throw new Error("electron_start_timeout");
  await recordOwnedProcesses();
  log(`ready vite_pid=${viteProc.pid} electron_pid=${electronProc.pid} url=${DEV_SERVER_URL}`);
}

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => {
    void cleanupOwnedFrontend(signal).finally(() => process.exit(0));
  });
}

process.on("uncaughtException", (error) => {
  console.error(`[HEBE][DEV_LAUNCHER][ERROR] ${error?.stack || error}`);
  void cleanupOwnedFrontend("uncaught_exception").finally(() => process.exit(1));
});

process.on("unhandledRejection", (error) => {
  console.error(`[HEBE][DEV_LAUNCHER][ERROR] ${error?.stack || error}`);
  void cleanupOwnedFrontend("unhandled_rejection").finally(() => process.exit(1));
});

if (require.main === module) {
  run().catch((error) => {
    console.error(`[HEBE][DEV_LAUNCHER][ERROR] ${error?.message || error}`);
    void cleanupOwnedFrontend("startup_failed").finally(() => process.exit(1));
  });
}

module.exports = { run };
