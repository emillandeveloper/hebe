import { useEffect, useMemo, useRef, useState } from "react";

type Device = MediaDeviceInfo;

export default function VtuberPreview() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [err, setErr] = useState<string>("");

  const videoInputs = useMemo(
    () => devices.filter((d) => d.kind === "videoinput"),
    [devices]
  );

  async function refreshDevices() {
    const list = await navigator.mediaDevices.enumerateDevices();
    setDevices(list);

    // Auto-select OBS Virtual Cam / VTube Studio si existe
    if (!deviceId) {
      const preferred = list.find(
        (d) =>
          d.kind === "videoinput" &&
          /obs|virtual|vtube|vts/i.test(d.label || "")
      );
      if (preferred?.deviceId) setDeviceId(preferred.deviceId);
      else {
        const first = list.find((d) => d.kind === "videoinput");
        if (first?.deviceId) setDeviceId(first.deviceId);
      }
    }
  }

  async function requestPermission() {
    setErr("");
    // Pide permiso una vez para que aparezcan los nombres (labels)
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach((t) => t.stop());
    await refreshDevices();
  }

  async function start() {
    setErr("");
    try {
      if (!deviceId) await requestPermission();

      const s = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
        audio: false,
      });

      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();
      }
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  function stop() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setStream(null);
    if (videoRef.current) videoRef.current.srcObject = null;
  }

  useEffect(() => {
    refreshDevices();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{
      borderRadius: 18,
      border: "1px solid rgba(255,255,255,0.12)",
      background: "rgba(0,0,0,0.16)",
      padding: 12
    }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap: 10 }}>
        <div style={{ fontWeight: 900 }}>VTuber Preview</div>
        <div style={{ display:"flex", gap: 8 }}>
          <button onClick={requestPermission} style={btn}>
            Permisos
          </button>
          <button onClick={start} style={btn}>
            Start
          </button>
          <button onClick={stop} style={{ ...btn, background:"rgba(255,77,77,0.14)" }}>
            Stop
          </button>
        </div>
      </div>

      <div style={{ marginTop: 10 }}>
        <select
          value={deviceId}
          onChange={(e) => setDeviceId(e.target.value)}
          style={select}
        >
          {videoInputs.length === 0 && <option value="">(Sin cámaras detectadas)</option>}
          {videoInputs.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || "(Nombre oculto: pulsa Permisos)"}
            </option>
          ))}
        </select>
      </div>

      <div style={{
        marginTop: 10,
        borderRadius: 16,
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.10)",
        background: "rgba(255,255,255,0.03)"
      }}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ width:"100%", display:"block", aspectRatio:"16/9", objectFit:"cover" }}
        />
      </div>

      {err && (
        <div style={{ marginTop: 10, color:"rgba(255,180,180,0.95)", fontSize: 12 }}>
          Error: {err}
        </div>
      )}

      <div style={{ marginTop: 8, color:"rgba(255,255,255,0.55)", fontSize: 12 }}>
        Tip: selecciona <b>OBS Virtual Camera</b> (o el virtual cam de VTube Studio si lo tienes).
      </div>
    </div>
  );
}

const btn: React.CSSProperties = {
  border: "1px solid rgba(255,255,255,0.14)",
  background: "rgba(255,255,255,0.06)",
  color: "rgba(255,255,255,0.9)",
  padding: "8px 10px",
  borderRadius: 12,
  cursor: "pointer",
  fontWeight: 700,
  fontSize: 12,
};

const select: React.CSSProperties = {
  width: "100%",
  padding: "10px 10px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.12)",
  background: "rgba(255,255,255,0.06)",
  color: "rgba(255,255,255,0.9)",
  outline: "none",
};
