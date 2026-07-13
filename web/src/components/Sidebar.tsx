import { useEffect, useState } from "react";
import { api } from "../api";
import type { ArtifactInfo, ModelInfo, UploadInfo } from "../types";

export interface RunConfig {
  modelId: string;
  lensName: string;
  prompt: string;
  imageNames: string[];
  layers: number[] | null;
  positions: string;
  topK: number;
  useChatTemplate: boolean;
  trackedConcepts: string[];
  device: string;
  dtype: string;
  force: boolean;
}

interface Props {
  busy: boolean;
  hasRun: boolean;
  onRun: (config: RunConfig) => void;
  onExportJson: () => void;
  onExportHtml: () => void;
}

export function Sidebar({ busy, hasRun, onRun, onExportJson, onExportHtml }: Props) {
  const [modelId, setModelId] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [device, setDevice] = useState("auto");
  const [dtype, setDtype] = useState("auto");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);
  const [lensName, setLensName] = useState("");
  const [prompt, setPrompt] = useState("The animal that spins webs has this many legs:");
  const [layersText, setLayersText] = useState("");
  const [positions, setPositions] = useState("all");
  const [topK, setTopK] = useState(10);
  const [useChatTemplate, setUseChatTemplate] = useState(false);
  const [trackText, setTrackText] = useState("");
  const [force, setForce] = useState(false);
  const [uploads, setUploads] = useState<UploadInfo[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    void refreshArtifacts();
  }, []);

  async function refreshArtifacts() {
    try {
      const list = await api.listArtifacts();
      setArtifacts(list);
      if (list.length > 0) {
        setLensName((current) => current || list[0].name);
      }
    } catch {
      /* server may not be up yet */
    }
  }

  async function loadModel() {
    setLoadingModel(true);
    setModelError(null);
    try {
      setModelInfo(await api.loadModel(modelId, device, dtype));
    } catch (exc) {
      setModelInfo(null);
      setModelError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoadingModel(false);
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!files) return;
    setUploadError(null);
    for (const file of Array.from(files)) {
      try {
        const info = await api.upload(file);
        setUploads((current) => [...current, info]);
      } catch (exc) {
        setUploadError(exc instanceof Error ? exc.message : String(exc));
      }
    }
  }

  function parseLayers(): number[] | null {
    const trimmed = layersText.trim();
    if (!trimmed || trimmed === "all") return null;
    return trimmed
      .split(",")
      .map((part) => parseInt(part.trim(), 10))
      .filter((n) => !Number.isNaN(n));
  }

  function run() {
    onRun({
      modelId,
      lensName,
      prompt,
      imageNames: uploads.map((u) => u.name),
      layers: parseLayers(),
      positions,
      topK,
      useChatTemplate,
      trackedConcepts: trackText
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      device,
      dtype,
      force,
    });
  }

  return (
    <div className="sidebar">
      <div>
        <label>Model ID</label>
        <input type="text" value={modelId} onChange={(e) => setModelId(e.target.value)} />
      </div>
      <div className="row2">
        <div>
          <label>Device</label>
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option value="auto">auto</option>
            <option value="cuda">cuda</option>
            <option value="mps">mps</option>
            <option value="cpu">cpu</option>
          </select>
        </div>
        <div>
          <label>Dtype</label>
          <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option value="auto">auto</option>
            <option value="float32">float32</option>
            <option value="bfloat16">bfloat16</option>
            <option value="float16">float16</option>
          </select>
        </div>
      </div>
      <button onClick={loadModel} disabled={loadingModel}>
        {loadingModel ? "Loading…" : "Load model"}
      </button>
      {modelError && <div className="error">{modelError}</div>}
      {modelInfo && (
        <div className="modelinfo">
          <b>{modelInfo.architecture}</b> ({modelInfo.kind}) —{" "}
          <span className={`status-${modelInfo.status}`}>{modelInfo.status}</span>
          <br />
          {modelInfo.n_layers} layers · hidden {modelInfo.hidden_size} · vocab{" "}
          {modelInfo.vocab_size}
          <br />
          {modelInfo.device} · {modelInfo.dtype} · {modelInfo.residual_location}
        </div>
      )}

      <div>
        <label>Lens artifact</label>
        <select value={lensName} onChange={(e) => setLensName(e.target.value)}>
          {artifacts.length === 0 && <option value="">(none found)</option>}
          {artifacts.map((a) => (
            <option key={a.name} value={a.name}>
              {a.name}
            </option>
          ))}
        </select>
        <button className="small" style={{ marginTop: 4 }} onClick={refreshArtifacts}>
          Refresh
        </button>
      </div>

      <div>
        <label>Prompt</label>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <div className="checkline">
          <input
            id="chat"
            type="checkbox"
            checked={useChatTemplate}
            onChange={(e) => setUseChatTemplate(e.target.checked)}
          />
          <label htmlFor="chat" style={{ margin: 0, textTransform: "none" }}>
            Apply chat template
          </label>
        </div>
      </div>

      <div>
        <label>Images (VLM only)</label>
        <input type="file" accept="image/*" multiple onChange={(e) => handleUpload(e.target.files)} />
        {uploads.map((u) => (
          <div key={u.name} className="muted">
            {u.name} ({u.width}×{u.height}){" "}
            <button
              className="small"
              onClick={() => setUploads(uploads.filter((x) => x.name !== u.name))}
            >
              ✕
            </button>
          </div>
        ))}
        {uploadError && <div className="error">{uploadError}</div>}
      </div>

      <div className="row2">
        <div>
          <label>Layers (blank = all)</label>
          <input
            type="text"
            placeholder="e.g. 0,4,8,12"
            value={layersText}
            onChange={(e) => setLayersText(e.target.value)}
          />
        </div>
        <div>
          <label>Top-K</label>
          <input
            type="number"
            min={1}
            max={100}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
          />
        </div>
      </div>
      <div>
        <label>Positions</label>
        <input
          type="text"
          placeholder="all | last:32 | 0,5,-1"
          value={positions}
          onChange={(e) => setPositions(e.target.value)}
        />
      </div>
      <div>
        <label>Pinned concepts (comma-separated)</label>
        <input
          type="text"
          placeholder="e.g. spider, legs"
          value={trackText}
          onChange={(e) => setTrackText(e.target.value)}
        />
      </div>
      <div className="checkline">
        <input id="force" type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
        <label htmlFor="force" style={{ margin: 0, textTransform: "none" }}>
          Force incompatible lens (warned)
        </label>
      </div>

      <button className="primary" onClick={run} disabled={busy || !lensName || !prompt}>
        {busy ? "Running…" : "Run"}
      </button>
      <div className="row2">
        <button onClick={onExportJson} disabled={!hasRun}>
          Export JSON
        </button>
        <button onClick={onExportHtml} disabled={!hasRun}>
          Export HTML
        </button>
      </div>
    </div>
  );
}
