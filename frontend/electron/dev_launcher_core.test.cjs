const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildElectronSpawnOptions,
  canTerminateVerifiedVite,
  decidePreflight,
  findVerifiedHebeElectron,
  isVerifiedHebeVite,
  ownershipRecordMatchesProcess,
} = require("./dev_launcher_core.cjs");

const workspace = String.raw`C:\Users\Leo Nifelheim\Documents\Hebe\hebe-ui\frontend`;
const token = "owner-token-123";

function vite(overrides = {}) {
  return {
    pid: 101,
    parentPid: 90,
    name: "node.exe",
    executablePath: String.raw`C:\Program Files\nodejs\node.exe`,
    commandLine: `node.exe "${workspace}\\node_modules\\vite\\bin\\vite.js" --port 5173 --mode hebe-dev-${token}`,
    startedAtMs: 10000,
    ...overrides,
  };
}

function electronRoot(overrides = {}) {
  return {
    pid: 201,
    parentPid: 190,
    name: "electron.exe",
    executablePath: `${workspace}\\node_modules\\electron\\dist\\electron.exe`,
    commandLine: `"${workspace}\\node_modules\\electron\\dist\\electron.exe" "${workspace}" --hebe-dev-token=${token}`,
    startedAtMs: 11000,
    ...overrides,
  };
}

function renderer(overrides = {}) {
  return {
    pid: 202,
    parentPid: 201,
    name: "electron.exe",
    executablePath: electronRoot().executablePath,
    commandLine: `electron.exe --type=renderer --app-path="${workspace}" http://localhost:5173/`,
    startedAtMs: 11100,
    ...overrides,
  };
}

function record(overrides = {}) {
  return {
    token,
    workspace,
    vite_pid: 101,
    vite_started_at_ms: 10000,
    electron_pid: 201,
    electron_started_at_ms: 11000,
    ...overrides,
  };
}

test("free port starts the canonical DEV runtime", () => {
  assert.deepEqual(
    decidePreflight({ listener: null, processes: [], record: null, workspace }),
    { action: "start", reason: "port_free" },
  );
});

test("verified same-workspace Vite and recorded Electron focus the existing instance", () => {
  const decision = decidePreflight({
    listener: vite(),
    processes: [vite(), electronRoot()],
    record: record(),
    workspace,
  });
  assert.equal(decision.action, "focus_existing");
  assert.equal(decision.electron.pid, 201);
});

test("legacy same-workspace Electron can be verified through its renderer", () => {
  assert.equal(findVerifiedHebeElectron([electronRoot(), renderer()], workspace)?.pid, 201);
  assert.equal(
    decidePreflight({ listener: vite(), processes: [vite(), electronRoot(), renderer()], record: null, workspace }).action,
    "focus_existing",
  );
});

test("verified same-workspace Vite without Electron is classified as stale", () => {
  const decision = decidePreflight({ listener: vite(), processes: [vite()], record: record(), workspace });
  assert.equal(decision.action, "cleanup_stale");
  assert.equal(decision.reason, "stale_hebe_vite");
  assert.equal(canTerminateVerifiedVite({ expected: decision.listener, current: vite(), workspace }), true);
});

test("foreign listener fails explicitly and is never classified for cleanup", () => {
  const foreign = vite({
    pid: 301,
    name: "python.exe",
    executablePath: String.raw`C:\Python\python.exe`,
    commandLine: "python.exe foreign_server.py --port 5173",
  });
  const decision = decidePreflight({ listener: foreign, processes: [], record: null, workspace });
  assert.equal(decision.action, "fail");
  assert.equal(decision.reason, "port_in_use_foreign_process");
  assert.equal(canTerminateVerifiedVite({ expected: foreign, current: foreign, workspace }), false);
});

test("a stale ownership record cannot turn PID reuse into Hebe ownership", () => {
  const reused = electronRoot({
    startedAtMs: 99000,
    commandLine: `"${workspace}\\node_modules\\electron\\dist\\electron.exe" "${workspace}" --hebe-dev-token=different`,
  });
  assert.equal(ownershipRecordMatchesProcess(record(), reused, { role: "electron", workspace }), false);
  assert.equal(findVerifiedHebeElectron([reused], workspace, record()), null);
});

test("a changed Vite start time blocks destructive cleanup even when the PID matches", () => {
  assert.equal(
    canTerminateVerifiedVite({ expected: vite(), current: vite({ startedAtMs: 99000 }), workspace }),
    false,
  );
});

test("Vite ownership requires the exact workspace Vite entry", () => {
  assert.equal(isVerifiedHebeVite(vite(), workspace), true);
  assert.equal(
    isVerifiedHebeVite(vite({ commandLine: String.raw`node C:\other\node_modules\vite\bin\vite.js --port 5173` }), workspace),
    false,
  );
});

test("Electron spawn keeps the GUI window visible on Windows", () => {
  const env = { ELECTRON_DEV: "1" };
  assert.deepEqual(buildElectronSpawnOptions({ workspace, env }), {
    cwd: workspace,
    env,
    stdio: "inherit",
    windowsHide: false,
  });
});
