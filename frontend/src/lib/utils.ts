export function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

export function fmtTime(tsSeconds: number) {
  const d = new Date(tsSeconds * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

export function uid() {
  return Math.random().toString(36).slice(2) + "-" + Date.now().toString(36);
}
