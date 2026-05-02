import { useEffect, useMemo, useRef, useState } from "react";

type Device = MediaDeviceInfo;

export default function VtuberPreview() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const workCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const [devices, setDevices] = useState<Device[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [err, setErr] = useState<string>("");

  const [zoom, setZoom] = useState<number>(1.42);
  const [offsetY, setOffsetY] = useState<number>(8);
  const [controlsOpen, setControlsOpen] = useState<boolean>(false);
  const [videoInfo, setVideoInfo] = useState<string>("");

  const videoInputs = useMemo(
    () => devices.filter((d) => d.kind === "videoinput"),
    [devices]
  );

  async function refreshDevices() {
    const list = await navigator.mediaDevices.enumerateDevices();
    setDevices(list);

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
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach((t) => t.stop());
    await refreshDevices();
  }

  async function start() {
    setErr("");

    try {
      if (!deviceId) await requestPermission();

      const videoConstraints: MediaTrackConstraints = deviceId
        ? {
            deviceId: { exact: deviceId },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 60 },
          }
        : {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 60 },
          };

      const s = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      setStream(s);

      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();

        const track = s.getVideoTracks()[0];
        const settings = track?.getSettings?.();
        if (settings?.width && settings?.height) {
          setVideoInfo(`${settings.width}×${settings.height}${settings.frameRate ? ` · ${Math.round(settings.frameRate)}fps` : ""}`);
        }
      }
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  function stop() {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (stream) stream.getTracks().forEach((t) => t.stop());
    setStream(null);
    setVideoInfo("");

    if (videoRef.current) videoRef.current.srcObject = null;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  useEffect(() => {
    refreshDevices();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (!stream) return;

    const draw = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const vw = video.videoWidth || 1920;
      const vh = video.videoHeight || 1080;

      if (!workCanvasRef.current) {
        workCanvasRef.current = document.createElement("canvas");
      }

      const work = workCanvasRef.current;
      if (work.width !== vw || work.height !== vh) {
        work.width = vw;
        work.height = vh;
      }

      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }

      const workCtx = work.getContext("2d", { willReadFrequently: true });
      const outCtx = canvas.getContext("2d");
      if (!workCtx || !outCtx) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      workCtx.clearRect(0, 0, vw, vh);
      workCtx.drawImage(video, 0, 0, vw, vh);

      const img = workCtx.getImageData(0, 0, vw, vh);
      removePureGreen(img.data);
      workCtx.putImageData(img, 0, 0);

      outCtx.clearRect(0, 0, vw, vh);
      outCtx.imageSmoothingEnabled = true;
      outCtx.imageSmoothingQuality = "high";
      outCtx.drawImage(work, 0, 0);

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [stream]);

  return (
    <div className="vtuberStage viewportStage">
      <video
        ref={videoRef}
        muted
        playsInline
        className="vtuberSourceVideo"
        onLoadedMetadata={() => {
          const v = videoRef.current;
          if (v?.videoWidth && v?.videoHeight) {
            setVideoInfo(`${v.videoWidth}×${v.videoHeight}`);
          }
        }}
      />

      <canvas
        ref={canvasRef}
        className="vtuberViewportVideo"
        style={{
          transform: `translate(-50%, calc(-50% + ${offsetY}%)) scale(${zoom})`,
        }}
      />

      {!stream && (
        <div className="vtuberEmpty">
          <div className="vtuberEmptyTitle">Hebe preview</div>
          <div className="vtuberEmptyText">Pulsa Start para mostrar la cámara virtual.</div>
        </div>
      )}

      {videoInfo && !controlsOpen && (
        <div className="vtuberQualityBadge">{videoInfo} · key #00FF00</div>
      )}

      <button
        className={"vtuberFab " + (controlsOpen ? "active" : "")}
        onClick={() => setControlsOpen((value) => !value)}
        title={controlsOpen ? "Ocultar controles" : "Mostrar controles"}
        aria-label={controlsOpen ? "Ocultar controles" : "Mostrar controles"}
      >
        ⚙
      </button>

      {controlsOpen && (
        <div className="vtuberOverlay viewportOverlay collapsible">
          <div className="vtuberOverlayHeader">
            <span>Preview {videoInfo ? `· ${videoInfo}` : ""}</span>
            <button
              className="miniIconBtn"
              onClick={() => setControlsOpen(false)}
              title="Cerrar controles"
              aria-label="Cerrar controles"
            >
              ×
            </button>
          </div>

          <div className="vtuberOverlayRow">
            <select
              className="select vtuberSelect"
              value={deviceId}
              onChange={(e) => setDeviceId(e.target.value)}
            >
              {videoInputs.length === 0 && <option value="">(Sin cámaras detectadas)</option>}
              {videoInputs.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || "(Nombre oculto: pulsa Permisos)"}
                </option>
              ))}
            </select>
          </div>

          <div className="vtuberOverlayRow compact">
            <button className="miniBtn" onClick={requestPermission}>Permisos</button>
            <button className="miniBtn" onClick={start}>Start</button>
            <button className="miniBtn danger" onClick={stop}>Stop</button>
            <span className="keyLabel fixedKey">Quita #00FF00</span>
          </div>

          <div className="viewportTuning">
            <label className="viewportField">
              <span>Zoom</span>
              <input
                type="range"
                min="1"
                max="2.5"
                step="0.01"
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
              />
            </label>

            <label className="viewportField">
              <span>Vertical</span>
              <input
                type="range"
                min="-35"
                max="35"
                step="1"
                value={offsetY}
                onChange={(e) => setOffsetY(Number(e.target.value))}
              />
            </label>
          </div>
        </div>
      )}

      {err && (
        <div className="vtuberError floating">
          Error: {err}
        </div>
      )}
    </div>
  );
}

/**
 * Quita solo el verde chroma #00FF00 y variantes cercanas.
 * Está ajustado para NO hacer key agresivo: menos mordiscos, más calidad.
 */
function removePureGreen(data: Uint8ClampedArray) {
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    const isPureChromaGreen =
      g > 170 &&
      r < 90 &&
      b < 90 &&
      g - r > 95 &&
      g - b > 95;

    const isDarkGreenEdge =
      g > 55 &&
      r < 42 &&
      b < 42 &&
      g - r > 25 &&
      g - b > 25;

    if (isPureChromaGreen || isDarkGreenEdge) {
      data[i + 3] = 0;
      continue;
    }

    // Despill conservador: solo reduce verdes raros muy dominantes.
    if (g > 115 && g > r + 42 && g > b + 42) {
      data[i + 1] = Math.max(r, b) + 22;
    }
  }
}
