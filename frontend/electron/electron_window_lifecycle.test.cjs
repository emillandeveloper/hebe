const test = require("node:test");
const assert = require("node:assert/strict");

const { focusMainWindow } = require("./electron_window_lifecycle.cjs");

test("second instance restores, shows, and focuses the existing window", () => {
  const calls = [];
  const window = {
    isDestroyed: () => false,
    isMinimized: () => true,
    restore: () => calls.push("restore"),
    show: () => calls.push("show"),
    focus: () => calls.push("focus"),
  };
  assert.equal(focusMainWindow(window), true);
  assert.deepEqual(calls, ["restore", "show", "focus"]);
});

test("second instance does nothing when no live window exists", () => {
  assert.equal(focusMainWindow(null), false);
  assert.equal(focusMainWindow({ isDestroyed: () => true }), false);
});
