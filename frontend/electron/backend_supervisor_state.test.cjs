const test = require("node:test");
const assert = require("node:assert/strict");

const {
  healthBelongsToManagedProcess,
  parseBackendHealth,
  reconcileBackendStatus,
} = require("./backend_supervisor_state.cjs");

test("health requires a successful Hebe health payload with a real pid", () => {
  assert.equal(parseBackendHealth({ statusCode: 200, body: '{"ok":true}' }), null);
  assert.equal(parseBackendHealth({ statusCode: 503, body: '{"ok":true,"pid":20}' }), null);
  assert.deepEqual(
    parseBackendHealth({ statusCode: 200, body: '{"ok":true,"pid":20,"parent_pid":10,"uptime_ms":1234}' }),
    { ok: true, pid: 20, parent_pid: 10, uptime_ms: 1234 },
  );
});

test("legacy health can be reconciled with the operating-system listener identity", () => {
  assert.deepEqual(
    parseBackendHealth(
      { statusCode: 200, body: '{"ok":true,"wake_loop":{"alive":true}}' },
      { pid: 30, parent_pid: 29, uptime_ms: 9000 },
    ),
    { ok: true, wake_loop: { alive: true }, pid: 30, parent_pid: 29, uptime_ms: 9000 },
  );
  assert.equal(
    parseBackendHealth(
      { statusCode: 200, body: '{"ok":true}' },
      { pid: 30, parent_pid: 29, uptime_ms: 9000 },
    ),
    null,
  );
});

test("uvicorn child health belongs to the managed Python launcher", () => {
  assert.equal(healthBelongsToManagedProcess({ pid: 20, parent_pid: 10 }, 10), true);
  assert.equal(healthBelongsToManagedProcess({ pid: 20, parent_pid: 9 }, 10), false);
});

test("old nonzero exit becomes historical after replacement is healthy", () => {
  const state = reconcileBackendStatus({
    health: { ok: true, pid: 222, parent_pid: 221, uptime_ms: 4567 },
    managedAlive: true,
    managedPid: 221,
    status: "failed",
    lastError: "Backend exited with code 1",
    historicalError: "Backend exited with code 1",
  });

  assert.equal(state.running, true);
  assert.equal(state.status, "healthy");
  assert.equal(state.pid, 222);
  assert.equal(state.supervisorPid, 221);
  assert.equal(state.managed, true);
  assert.equal(state.lastError, "");
  assert.equal(state.historicalError, "Backend exited with code 1");
  assert.equal(state.uptimeMs, 4567);
});

test("a live launcher without healthy runtime is not reported running", () => {
  const state = reconcileBackendStatus({
    health: null,
    managedAlive: true,
    managedPid: 77,
    managedStartTime: 1000,
    now: 2000,
    status: "starting",
  });
  assert.equal(state.running, false);
  assert.equal(state.status, "starting");
  assert.equal(state.pid, null);
  assert.equal(state.supervisorPid, 77);
});
