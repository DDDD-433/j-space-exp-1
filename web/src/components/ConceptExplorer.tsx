import { useMemo, useState } from "react";
import { api } from "../api";
import type { RunConfig } from "./Sidebar";
import type { ConceptEntry, DecompositionRecord, RunResult, SelectedCell } from "../types";

interface Props {
  run: RunResult;
  selected: SelectedCell | null;
  config: RunConfig | null;
  onPinConcept: (concept: string) => void;
}

function isWhitespaceOnly(text: string): boolean {
  return text.trim().length === 0;
}

function isPunctuationOnly(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.length > 0 && !/[\p{L}\p{N}]/u.test(trimmed);
}

export function ConceptExplorer({ run, selected, config, onPinConcept }: Props) {
  const [hideWhitespace, setHideWhitespace] = useState(false);
  const [hidePunctuation, setHidePunctuation] = useState(false);
  const [collapseVariants, setCollapseVariants] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [decomposition, setDecomposition] = useState<DecompositionRecord | null>(null);
  const [decomposing, setDecomposing] = useState(false);
  const [decomposeK, setDecomposeK] = useState(10);
  const [decomposeError, setDecomposeError] = useState<string | null>(null);

  const cell = useMemo(() => {
    if (!selected) return null;
    return (
      run.cells.find((c) => c.layer === selected.layer && c.position === selected.position) ?? null
    );
  }, [run, selected]);

  const filtered = useMemo(() => {
    if (!cell) return [];
    let entries = cell.jlens_top;
    if (hideWhitespace) entries = entries.filter((e) => !isWhitespaceOnly(e.token_text));
    if (hidePunctuation) entries = entries.filter((e) => !isPunctuationOnly(e.token_text));
    if (!collapseVariants) return entries.map((e) => ({ entry: e, variants: [] as ConceptEntry[] }));
    // Group tokenizer variants (case/leading-space) under the best-scoring one.
    const groups = new Map<string, { entry: ConceptEntry; variants: ConceptEntry[] }>();
    for (const entry of entries) {
      const key = entry.token_text.trim().toLowerCase();
      const existing = groups.get(key);
      if (!existing) groups.set(key, { entry, variants: [] });
      else existing.variants.push(entry);
    }
    return [...groups.values()];
  }, [cell, hideWhitespace, hidePunctuation, collapseVariants]);

  async function decompose() {
    if (!selected || !config || cell?.is_model_output) return;
    setDecomposing(true);
    setDecomposeError(null);
    try {
      setDecomposition(
        await api.decompose({
          model_id: config.modelId,
          lens_name: config.lensName,
          prompt: config.prompt,
          image_names: config.imageNames,
          layer: selected.layer,
          position: selected.position,
          k: decomposeK,
          use_chat_template: config.useChatTemplate,
        }),
      );
    } catch (exc) {
      setDecomposeError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setDecomposing(false);
    }
  }

  if (!selected || !cell) {
    return <div className="empty">Select a cell in the Lens Grid to explore its concepts.</div>;
  }

  const info = run.positions.find((p) => p.index === selected.position);

  return (
    <div className="cols">
      <div>
        <div className="panel">
          <h3>
            Top J-lens concepts — L{selected.layer} · pos {selected.position}
            {cell.is_model_output ? " (model output distribution)" : ""}
          </h3>
          <div className="muted">
            source token <span className="tok">{info?.token_text ?? ""}</span> · {info?.modality} ·
            ‖h‖ = {cell.activation_norm.toFixed(2)} · ‖Jh‖ = {cell.transported_norm.toFixed(2)}
          </div>
          <div style={{ margin: "8px 0" }}>
            <span
              className={`pill ${hideWhitespace ? "active" : ""}`}
              onClick={() => setHideWhitespace(!hideWhitespace)}
            >
              hide whitespace
            </span>
            <span
              className={`pill ${hidePunctuation ? "active" : ""}`}
              onClick={() => setHidePunctuation(!hidePunctuation)}
            >
              hide punctuation
            </span>
            <span
              className={`pill ${collapseVariants ? "active" : ""}`}
              onClick={() => setCollapseVariants(!collapseVariants)}
            >
              collapse variants
            </span>
            <span className={`pill ${showRaw ? "active" : ""}`} onClick={() => setShowRaw(!showRaw)}>
              raw tokens
            </span>
          </div>
          <table className="detail">
            <thead>
              <tr>
                <th>#</th>
                <th>concept</th>
                <th>token id</th>
                <th>lens score</th>
                <th>norm.</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(({ entry, variants }, i) => (
                <tr key={`${entry.token_id}-${i}`}>
                  <td>{i + 1}</td>
                  <td>
                    <span className="tok">
                      {showRaw ? JSON.stringify(entry.token_text) : entry.token_display}
                    </span>
                    {variants.length > 0 && (
                      <span className="muted"> +{variants.length} variants</span>
                    )}
                  </td>
                  <td>{entry.token_id}</td>
                  <td>{entry.score.toFixed(3)}</td>
                  <td>{entry.normalized_score.toFixed(2)}</td>
                  <td>
                    <button className="small" onClick={() => onPinConcept(entry.token_text)}>
                      pin
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="muted" style={{ marginTop: 6 }}>
            Raw tokenizer output is always preserved — toggle “raw tokens” to see it.
          </div>
        </div>
      </div>
      <div>
        <div className="panel">
          <h3>Sparse J-space decomposition</h3>
          <div className="muted">
            Non-negative pursuit of ≤ K J-lens directions approximating this activation. Not the
            same as top-K by score.
          </div>
          <div style={{ display: "flex", gap: 8, margin: "8px 0", alignItems: "center" }}>
            <label className="muted" style={{ margin: 0 }}>
              K
            </label>
            <input
              type="number"
              min={1}
              max={64}
              value={decomposeK}
              onChange={(e) => setDecomposeK(parseInt(e.target.value, 10) || 10)}
              style={{
                width: 64,
                background: "var(--panel2)",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: "4px 6px",
              }}
            />
            <button onClick={decompose} disabled={decomposing || cell.is_model_output || !config}>
              {decomposing ? "Solving…" : "Decompose"}
            </button>
          </div>
          {cell.is_model_output && (
            <div className="muted">Decomposition targets lens layers, not the output row.</div>
          )}
          {decomposeError && <div className="error">{decomposeError}</div>}
          {decomposition &&
            decomposition.layer === selected.layer &&
            decomposition.position === selected.position && (
              <>
                <table className="detail">
                  <thead>
                    <tr>
                      <th>concept</th>
                      <th>token id</th>
                      <th>coefficient</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decomposition.entries.map((e, i) => (
                      <tr key={i}>
                        <td>
                          <span className="tok">{e.token_display}</span>
                        </td>
                        <td>{e.token_id}</td>
                        <td>{e.coefficient.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <dl className="kv" style={{ marginTop: 8 }}>
                  <dt>reconstruction error</dt>
                  <dd>{decomposition.reconstruction_error.toFixed(4)}</dd>
                  <dt>residual norm</dt>
                  <dd>{decomposition.residual_norm.toFixed(4)}</dd>
                  <dt>explained norm fraction</dt>
                  <dd>{(decomposition.explained_norm_fraction * 100).toFixed(1)}%</dd>
                  <dt>atoms used</dt>
                  <dd>
                    {decomposition.n_iterations} / {decomposition.k_requested}
                  </dd>
                </dl>
                <div className="muted" style={{ marginTop: 6 }}>
                  ⚠ {decomposition.warning}
                </div>
              </>
            )}
        </div>
      </div>
    </div>
  );
}
