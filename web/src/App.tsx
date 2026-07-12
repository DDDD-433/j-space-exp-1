import { useCallback, useMemo, useState } from "react";
import { api } from "./api";
import { Sidebar, type RunConfig } from "./components/Sidebar";
import { LensGrid } from "./components/LensGrid";
import { ConceptExplorer } from "./components/ConceptExplorer";
import { RankTracking } from "./components/RankTracking";
import { LensComparison } from "./components/LensComparison";
import { ImageTokens } from "./components/ImageTokens";
import { MetadataTab } from "./components/MetadataTab";
import type { RunResult, SelectedCell } from "./types";

const TABS = [
  "Lens Grid",
  "Concept Explorer",
  "Rank Tracking",
  "Image Tokens",
  "Lens Comparison",
  "Metadata",
] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [run, setRun] = useState<RunResult | null>(null);
  const [config, setConfig] = useState<RunConfig | null>(null);
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const [tab, setTab] = useState<Tab>("Lens Grid");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runInspect = useCallback(async (cfg: RunConfig) => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.inspect({
        model_id: cfg.modelId,
        lens_name: cfg.lensName,
        prompt: cfg.prompt,
        image_names: cfg.imageNames,
        layers: cfg.layers,
        positions: cfg.positions,
        top_k: cfg.topK,
        use_chat_template: cfg.useChatTemplate,
        tracked_concepts: cfg.trackedConcepts,
        device: cfg.device,
        dtype: cfg.dtype,
        force: cfg.force,
      });
      setRun(result);
      setConfig(cfg);
      setSelected(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }, []);

  const rerunWithConcepts = useCallback(
    (concepts: string[]) => {
      if (!config) return;
      const merged = Array.from(new Set([...config.trackedConcepts, ...concepts]));
      void runInspect({ ...config, trackedConcepts: merged });
    },
    [config, runInspect],
  );

  const exportJson = useCallback(() => {
    if (!run) return;
    const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `openjspace-run-${run.metadata.run_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [run]);

  const exportHtml = useCallback(() => {
    if (!run) return;
    window.open(`/api/runs/${run.metadata.run_id}/report.html`, "_blank");
  }, [run]);

  const hasImageTokens = useMemo(
    () => run?.positions.some((p) => p.modality === "image_token") ?? false,
    [run],
  );

  return (
    <>
      <div className="topbar">
        <h1>OpenJSpace</h1>
        <span className="sub">
          Jacobian-lens &amp; J-space visualizer — approximate verbalizable content, not literal
          thoughts
        </span>
      </div>
      <div className="layout">
        <Sidebar
          busy={busy}
          onRun={runInspect}
          onExportJson={exportJson}
          onExportHtml={exportHtml}
          hasRun={run !== null}
        />
        <div className="main">
          {run && <div className="disclaimerbar">{run.metadata.disclaimer}</div>}
          {run && run.metadata.warnings.length > 0 && (
            <div className="warningbar">
              {run.metadata.warnings.map((w, i) => (
                <div key={i}>⚠ {w}</div>
              ))}
            </div>
          )}
          {error && <div className="warningbar error">✗ {error}</div>}
          <div className="tabs">
            {TABS.map((t) => (
              <button
                key={t}
                className={t === tab ? "active" : ""}
                onClick={() => setTab(t)}
                disabled={t === "Image Tokens" && !hasImageTokens && run !== null}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="tabbody">
            {!run && (
              <div className="empty">
                Load a model and a fitted lens, enter a prompt, then press <b>Run</b>.
                <br />
                Fit a lens first with <code>openjspace fit</code> if none is listed.
              </div>
            )}
            {run && tab === "Lens Grid" && (
              <LensGrid run={run} selected={selected} onSelect={setSelected} />
            )}
            {run && tab === "Concept Explorer" && (
              <ConceptExplorer
                run={run}
                selected={selected}
                config={config}
                onPinConcept={(c) => rerunWithConcepts([c])}
              />
            )}
            {run && tab === "Rank Tracking" && (
              <RankTracking run={run} selected={selected} onSelect={setSelected} />
            )}
            {run && tab === "Image Tokens" && (
              <ImageTokens run={run} selected={selected} onSelect={setSelected} />
            )}
            {run && tab === "Lens Comparison" && <LensComparison run={run} selected={selected} />}
            {run && tab === "Metadata" && <MetadataTab run={run} />}
          </div>
        </div>
      </div>
    </>
  );
}
