function numericPid(value) {
  const pid = Number(value);
  return Number.isInteger(pid) && pid > 0 ? pid : null;
}

function parseBackendHealth(response, fallbackIdentity = null) {
  if (!response || Number(response.statusCode) < 200 || Number(response.statusCode) >= 300) {
    return null;
  }
  try {
    const payload = JSON.parse(String(response.body || ""));
    const nativePid = numericPid(payload.pid);
    const fallbackPid = numericPid(fallbackIdentity?.pid);
    if (nativePid === null && fallbackPid !== null && !("wake_loop" in payload)) return null;
    const pid = nativePid || fallbackPid;
    if (payload.ok !== true || pid === null) return null;
    return {
      ...payload,
      pid,
      parent_pid: numericPid(payload.parent_pid) || numericPid(fallbackIdentity?.parent_pid),
      uptime_ms: Math.max(0, Number(payload.uptime_ms) || Number(fallbackIdentity?.uptime_ms) || 0),
    };
  } catch (_) {
    return null;
  }
}

function healthBelongsToManagedProcess(health, managedPid) {
  const ownerPid = numericPid(managedPid);
  if (!health || ownerPid === null) return false;
  return health.pid === ownerPid || health.parent_pid === ownerPid;
}

function reconcileBackendStatus({
  health,
  managedAlive = false,
  managedPid = null,
  managedStartTime = 0,
  status = "stopped",
  lastRestartTime = null,
  lastError = "",
  historicalError = "",
  now = Date.now(),
} = {}) {
  if (health && health.ok === true && numericPid(health.pid) !== null) {
    return {
      running: true,
      pid: numericPid(health.pid),
      supervisorPid: managedAlive ? numericPid(managedPid) : null,
      managed: healthBelongsToManagedProcess(health, managedPid),
      uptimeMs: Math.max(0, Number(health.uptime_ms) || 0),
      lastRestartTime,
      lastError: "",
      historicalError: String(historicalError || lastError || ""),
      status: "healthy",
    };
  }

  const transitional = new Set(["starting", "restarting", "stopping"]);
  const resolvedStatus = managedAlive && transitional.has(status)
    ? status
    : status === "failed"
      ? "failed"
      : "stopped";
  return {
    running: false,
    pid: null,
    supervisorPid: managedAlive ? numericPid(managedPid) : null,
    managed: false,
    uptimeMs: managedAlive && managedStartTime ? Math.max(0, now - managedStartTime) : 0,
    lastRestartTime,
    lastError: resolvedStatus === "failed" ? String(lastError || "") : "",
    historicalError: String(historicalError || ""),
    status: resolvedStatus,
  };
}

module.exports = {
  healthBelongsToManagedProcess,
  parseBackendHealth,
  reconcileBackendStatus,
};
