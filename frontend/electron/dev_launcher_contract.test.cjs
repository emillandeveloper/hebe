const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const config = require("./dev_runtime_config.cjs");

const frontend = path.resolve(__dirname, "..");
const read = (relative) => fs.readFileSync(path.join(frontend, relative), "utf8");

test("Vite, launcher, and Electron share one canonical DEV endpoint", () => {
  assert.equal(config.DEV_SERVER_PORT, 5173);
  assert.equal(config.DEV_SERVER_URL, "http://localhost:5173");
  const viteConfig = read("vite.config.ts");
  const main = read("electron/main.cjs");
  const launcher = read("electron/dev_launcher.cjs");
  assert.match(viteConfig, /strictPort:\s*true/);
  assert.match(viteConfig, /dev_runtime_config\.cjs/);
  assert.match(main, /loadURL\(DEV_SERVER_URL\)/);
  assert.match(launcher, /DEV_SERVER_URL/);
});

test("electron:dev uses the ownership-aware launcher without wait-on or concurrently", () => {
  const packageJson = JSON.parse(read("package.json"));
  assert.equal(packageJson.scripts["electron:dev"], "node electron/dev_launcher.cjs");
  assert.doesNotMatch(packageJson.scripts["electron:dev"], /wait-on|concurrently/);
});

test("launcher removes ELECTRON_RUN_AS_NODE instead of passing an empty value", () => {
  const launcher = read("electron/dev_launcher.cjs");
  assert.match(launcher, /delete electronEnv\.ELECTRON_RUN_AS_NODE/);
  assert.doesNotMatch(launcher, /ELECTRON_RUN_AS_NODE:\s*["']{2}/);
});

test("Electron acquires a single-instance lock before installing runtime lifecycle", () => {
  const main = read("electron/main.cjs");
  assert.ok(main.indexOf("requestSingleInstanceLock") < main.indexOf("app.whenReady"));
  assert.match(main, /app\.on\("second-instance"/);
  assert.match(main, /focusMainWindow\(mainWindow\)/);
});

test("launcher can restore a hidden Electron window by enumerating windows for its PID", () => {
  const launcher = read("electron/dev_launcher.cjs");
  assert.match(launcher, /EnumWindows/);
  assert.match(launcher, /GetWindowThreadProcessId/);
  assert.match(launcher, /GetWindowTextLength/);
  assert.match(launcher, /IsWindowVisible/);
  assert.match(launcher, /!IsIconic\(match\)/);
});

test("Electron records the DEV URL actually loaded by its renderer", () => {
  const main = read("electron/main.cjs");
  assert.match(main, /did-finish-load/);
  assert.match(main, /webContents\.getURL\(\)/);
});

test("normal frontend shutdown only stops a backend owned by Electron", () => {
  const main = read("electron/main.cjs");
  assert.match(main, /shouldStopBackendOnQuit\(\{ ownedByElectron: backendOwnedByElectron \}\)/);
  assert.doesNotMatch(main, /window-all-closed"[\s\S]{0,100}killBackendTree/);
});
