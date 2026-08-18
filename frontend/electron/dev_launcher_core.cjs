const path = require("path");

function canonicalPath(value) {
  return path.resolve(String(value || "")).replaceAll("/", "\\").toLowerCase();
}

function commandContainsPath(commandLine, expectedPath) {
  const command = String(commandLine || "").replaceAll("/", "\\").toLowerCase();
  return Boolean(command && command.includes(canonicalPath(expectedPath)));
}

function validPid(value) {
  const pid = Number(value);
  return Number.isInteger(pid) && pid > 0 ? pid : null;
}

function sameStartTime(left, right, toleranceMs = 1500) {
  const a = Number(left);
  const b = Number(right);
  return Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0 && Math.abs(a - b) <= toleranceMs;
}

function isVerifiedHebeVite(processInfo, workspace) {
  if (!processInfo || validPid(processInfo.pid) === null) return false;
  if (String(processInfo.name || "").toLowerCase() !== "node.exe") return false;
  return commandContainsPath(
    processInfo.commandLine,
    path.join(workspace, "node_modules", "vite", "bin", "vite.js"),
  );
}

function ownershipRecordMatchesProcess(record, processInfo, { role, workspace } = {}) {
  if (!record || !processInfo || canonicalPath(record.workspace) !== canonicalPath(workspace)) return false;
  const prefix = role === "electron" ? "electron" : "vite";
  if (validPid(record[`${prefix}_pid`]) !== validPid(processInfo.pid)) return false;
  if (!sameStartTime(record[`${prefix}_started_at_ms`], processInfo.startedAtMs)) return false;
  if (!commandContainsPath(processInfo.commandLine, workspace)) return false;
  const token = String(record.token || "").trim();
  return Boolean(token && String(processInfo.commandLine || "").includes(token));
}

function findVerifiedHebeElectron(processes, workspace, record = null) {
  const items = Array.isArray(processes) ? processes : [];
  if (record) {
    const recorded = items.find((item) => validPid(item.pid) === validPid(record.electron_pid));
    if (ownershipRecordMatchesProcess(record, recorded, { role: "electron", workspace })) {
      return recorded;
    }
  }

  const renderer = items.find((item) => (
    String(item.name || "").toLowerCase() === "electron.exe"
    && String(item.commandLine || "").includes("--type=renderer")
    && commandContainsPath(item.commandLine, workspace)
  ));
  if (!renderer) return null;
  const root = items.find((item) => validPid(item.pid) === validPid(renderer.parentPid));
  if (!root || String(root.name || "").toLowerCase() !== "electron.exe") return null;
  return commandContainsPath(root.executablePath || root.commandLine, workspace) ? root : null;
}

function decidePreflight({ listener, processes, record, workspace }) {
  if (!listener) return { action: "start", reason: "port_free" };
  if (!isVerifiedHebeVite(listener, workspace)) {
    return {
      action: "fail",
      reason: "port_in_use_foreign_process",
      listener,
    };
  }
  const electron = findVerifiedHebeElectron(processes, workspace, record);
  if (electron) {
    return {
      action: "focus_existing",
      reason: "existing_hebe_dev_instance",
      listener,
      electron,
    };
  }
  return {
    action: "cleanup_stale",
    reason: "stale_hebe_vite",
    listener,
  };
}

function canTerminateVerifiedVite({ expected, current, workspace }) {
  return Boolean(
    expected
    && current
    && validPid(expected.pid) === validPid(current.pid)
    && sameStartTime(expected.startedAtMs, current.startedAtMs)
    && isVerifiedHebeVite(current, workspace)
  );
}

function buildElectronSpawnOptions({ workspace, env }) {
  return {
    cwd: workspace,
    env,
    stdio: "inherit",
    windowsHide: false,
  };
}

module.exports = {
  buildElectronSpawnOptions,
  canonicalPath,
  canTerminateVerifiedVite,
  decidePreflight,
  findVerifiedHebeElectron,
  isVerifiedHebeVite,
  ownershipRecordMatchesProcess,
};
