import { useMemo } from "react";
import type { CellRecord, ConceptEntry, RunResult, SelectedCell } from "../types";

interface Props {
  run: RunResult;
  selected: SelectedCell | null;
}

function TopList({ title, entries }: { title: string; entries: ConceptEntry[] }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      <table className="detail">
        <thead>
          <tr>
            <th>#</th>
            <th>concept</th>
            <th>score</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>
                <span className="tok">{e.token_display}</span>
              </td>
              <td>{e.score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LensComparison({ run, selected }: Props) {
  const positions = useMemo(
    () => [...new Set(run.cells.map((c) => c.position))].sort((a, b) => a - b),
    [run],
  );
  const position = selected?.position ?? positions[positions.length - 1];

  const cellsAt = useMemo(() => {
    const map = new Map<number, CellRecord>();
    for (const cell of run.cells) if (cell.position === position) map.set(cell.layer, cell);
    return map;
  }, [run, position]);

  const lensLayers = run.metadata.layers.filter((l) => !cellsAt.get(l)?.is_model_output);
  const outputLayer = run.metadata.layers.find((l) => cellsAt.get(l)?.is_model_output);
  const cell = selected ? cellsAt.get(selected.layer) : cellsAt.get(lensLayers[lensLayers.length - 1]);
  const outputCell = outputLayer !== undefined ? cellsAt.get(outputLayer) : undefined;

  // Divergence across layers: per-layer overlap and rank correlation at this position.
  const divergence = lensLayers.map((layer) => {
    const c = cellsAt.get(layer);
    return {
      layer,
      overlap: c?.comparison["topk_overlap"] ?? 0,
      corr: c?.comparison["rank_correlation"] ?? 0,
    };
  });

  if (!cell) {
    return <div className="empty">Select a cell in the Lens Grid first.</div>;
  }

  return (
    <div>
      <div className="muted" style={{ marginBottom: 10 }}>
        Position {position} · layer L{cell.layer}. No method is universally “correct”: the J-lens is
        a corpus-averaged linear transport, the logit lens assumes shared coordinates across
        layers, and the final output describes the model's next-token distribution.
      </div>
      <div className="cols">
        <TopList title={`J-lens — L${cell.layer}`} entries={cell.jlens_top} />
        <TopList title={`Logit lens — L${cell.layer}`} entries={cell.logit_lens_top} />
        {outputCell && (
          <TopList title="Model output distribution (final layer)" entries={outputCell.jlens_top} />
        )}
      </div>
      <div className="panel">
        <h3>J-lens vs. logit-lens divergence across layers — position {position}</h3>
        <table className="detail">
          <thead>
            <tr>
              <th>layer</th>
              <th>top-{run.metadata.top_k} overlap</th>
              <th>rank correlation (top-set union)</th>
            </tr>
          </thead>
          <tbody>
            {divergence.map((d) => (
              <tr key={d.layer}>
                <td>L{d.layer}</td>
                <td>
                  <div
                    style={{
                      display: "inline-block",
                      width: 120,
                      height: 8,
                      background: "var(--panel2)",
                      borderRadius: 4,
                      marginRight: 8,
                      verticalAlign: "middle",
                    }}
                  >
                    <div
                      style={{
                        width: `${(d.overlap * 100).toFixed(0)}%`,
                        height: "100%",
                        background: "var(--accent)",
                        borderRadius: 4,
                      }}
                    />
                  </div>
                  {(d.overlap * 100).toFixed(0)}%
                </td>
                <td>{d.corr.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="muted" style={{ marginTop: 6 }}>
          The two lenses typically agree in the last few layers (residual connections dominate) and
          diverge earlier, where the logit lens's shared-coordinates assumption degrades.
        </div>
      </div>
    </div>
  );
}
