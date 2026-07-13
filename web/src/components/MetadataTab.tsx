import type { RunResult } from "../types";

export function MetadataTab({ run }: { run: RunResult }) {
  const meta = run.metadata;
  const lens = meta.lens_metadata as Record<string, unknown>;
  return (
    <div className="cols">
      <div>
        <div className="panel">
          <h3>Run</h3>
          <dl className="kv">
            <dt>run id</dt>
            <dd>{meta.run_id}</dd>
            <dt>created</dt>
            <dd>{meta.created_at}</dd>
            <dt>model</dt>
            <dd>{meta.model_id}</dd>
            <dt>architecture</dt>
            <dd>{meta.model_architecture}</dd>
            <dt>kind</dt>
            <dd>{meta.model_kind}</dd>
            <dt>device / dtype</dt>
            <dd>
              {meta.device} / {meta.dtype}
            </dd>
            <dt>chat template</dt>
            <dd>{String(meta.used_chat_template)}</dd>
            <dt>layers</dt>
            <dd>{meta.layers.join(", ")}</dd>
            <dt>positions</dt>
            <dd>{meta.n_positions}</dd>
            <dt>top-K</dt>
            <dd>{meta.top_k}</dd>
            <dt>patch mapping</dt>
            <dd>{meta.patch_mapping}</dd>
          </dl>
        </div>
        <div className="panel">
          <h3>Prompt</h3>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, margin: 0 }}>{meta.prompt}</pre>
        </div>
      </div>
      <div>
        <div className="panel">
          <h3>Lens artifact</h3>
          <dl className="kv">
            {Object.entries(lens).map(([key, value]) => (
              <span key={key} style={{ display: "contents" }}>
                <dt>{key}</dt>
                <dd>{typeof value === "object" ? JSON.stringify(value) : String(value)}</dd>
              </span>
            ))}
          </dl>
        </div>
        <div className="panel">
          <h3>Scientific note</h3>
          <div className="muted">{meta.disclaimer}</div>
          {meta.warnings.length > 0 && (
            <ul className="muted">
              {meta.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
