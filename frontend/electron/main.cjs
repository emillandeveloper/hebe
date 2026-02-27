const { app, BrowserWindow, session } = require("electron");
const path = require("path");
const { spawn, execFileSync } = require("child_process");
const http = require("http");
const fs = require("fs");

let backendProc = null;

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

function httpGet(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve(res.statusCode || 0);
    });
    req.on("error", reject);
  });
}

async function waitForBackend(url, tries = 80, delayMs = 150) {
  for (let i = 0; i < tries; i++) {
    try {
      const code = await httpGet(url);
      if (code >= 200 && code < 500) return true;
    } catch (_) {}
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return false;
}

function startBackend() {
  const backendDir = path.resolve(__dirname, "../../backend");
  const py = path.join(backendDir, ".venv", "Scripts", "python.exe");

  console.log("[backend] backendDir:", backendDir);
  console.log("[backend] python:", py, "exists:", fs.existsSync(py));

  if (!fs.existsSync(py)) {
    console.error("[backend] No existe el python del venv. ¿Creaste la venv en backend?");
    return;
  }

  backendProc = spawn(
    py, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"], {
    cwd: backendDir,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
  });

  backendProc.on("error", (e) => console.error("[backend] spawn error:", e));
  backendProc.stdout.on("data", (d) => console.log("[backend]", d.toString().trim()));
  backendProc.stderr.on("data", (d) => console.log("[backend:err]", d.toString().trim()));
}

function killBackendTree() {
  if (!backendProc || !backendProc.pid) return;
  const pid = backendProc.pid;

  try {
    // mata proceso + hijos (clave si uvicorn/reload spawnea)
    execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore" });
  } catch (_) {
    // si taskkill falla, intenta kill normal
    try { backendProc.kill(); } catch (_) {}
  } finally {
    backendProc = null;
  }
}

async function createWindow() {
  allowMediaPermissions();

  // Arranca backend (solo en dev / si quieres siempre)
  startBackend();

  // Espera a backend antes de abrir UI (evita “Sin conexión”)
  const ok = await waitForBackend("http://127.0.0.1:8000/health", 360, 250);
  console.log("[backend] health:", ok ? "OK" : "NOT READY");

  const win = new BrowserWindow({
    width: 1200,
    height: 760,
    backgroundColor: "#070916",
    webPreferences: { contextIsolation: true },
  });

  if (process.env.ELECTRON_DEV === "1") {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

app.whenReady().then(createWindow);

// cierres “limpios”
app.on("before-quit", () => killBackendTree());
app.on("window-all-closed", () => {
  killBackendTree();
  app.quit();
});
