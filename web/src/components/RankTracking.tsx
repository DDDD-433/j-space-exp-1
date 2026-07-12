import { useMemo, useState } from "react";
import type { RunResult, SelectedCell, TrackedConcept } from "../types";

interface Props {
  run: RunResult;
  selected: SelectedCell | null;
  onSelect: (cell: SelectedCell) => void;
}

const COLORS = ["#6ea8fe", "#59c9a5", "#f0b429", "#f26d6d", "#b48ef0", "#6ee7f0"];

function LineChart({
  series,
  xLabels,
  yMax,
  width = 560,
  height = 220,
  yLabel,
  invertY,
}: {
  series: { name: string; color: string; values: number[] }[];
  xLabels: (string | number)[];
  yMax: number;
  width?: number;
  height?: number;
  yLabel: string;
  invertY?: boolean;
}) {
  const pad = { left: 46, right: 10, top: 10, bottom: 26 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const n = xLabels.length;
  const x = (i: number) => pad.left + (n <= 1 ? 0 : (i / (n - 1)) * innerW);
  const y = (v: number) => {
    const t = Math.max(0, Math.min(1, v / yMax));
    return pad.top + (invertY ? t : 1 - t) * innerH;
  };
  return (
    <svg className="chart" width={width} height={height}>
      <text x={12} y={height / 2} fill="var(--muted)" fontSize={10} transform={`rotate(-90 12 ${height / 2})`} textAnchor="middle">
        {yLabel}
      </text>
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line
            x1={pad.left}
            x2={width - pad.right}
            y1={pad.top + f * innerH}
            y2={pad.top + f * innerH}
            stroke="var(--border)"
          />
          <text x={pad.left - 6} y={pad.top + f * innerH + 3} fill="var(--muted)" fontSize={9} textAnchor="end">
            {invertY ? Math.round(f * yMax) : Math.round((1 - f) * yMax)}
          </text>
        </g>
      ))}
      {xLabels.map((label, i) =>
        n <= 24 || i % Math.ceil(n / 24) === 0 ? (
          <text key={i} x={x(i)} y={height - 8} fill="var(--muted)" fontSize={9} textAnchor="middle">
            {label}
          </text>
        ) : null,
      )}
      {series.map((s) => (
        <polyline
          key={s.name}
          fill="none"
          stroke={s.color}
          strokeWidth={1.8}
          points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
        />
      ))}
    </svg>
  );
}

export function RankTracking({ run, selected, onSelect }: Props) {
  const [heatmapToken, setHeatmapToken] = useState<number | null>(null);

  const layers = run.metadata.layers;
  const positions = useMemo(
    () => [...new Set(run.cells.map((c) => c.position))].sort((a, b) => a - b),
    [run],
  );
  const tracked = run.tracked;

  if (tracked.length === 0) {
    return (
      <div className="empty">
        No pinned concepts. Add concepts in the sidebar (“Pinned concepts”) or pin one from the
        Concept Explorer, then re-run.
      </div>
    );
  }

  const vocabSize = Number(run.metadata.lens_metadata["vocab_size"] ?? 50000);
  const logMax = Math.log10(vocabSize);
  const selectedCol = selected ? positions.indexOf(selected.position) : positions.length - 1;
  const col = selectedCol >= 0 ? selectedCol : positions.length - 1;
  const heatToken: TrackedConcept =
    tracked.find((t) => t.token_id === heatmapToken) ?? tracked[0];

  return (
    <div>
      <div className="panel">
        <h3>
          Rank across layers — position {positions[col]}{" "}
          <span className="muted">(log10 rank, 0 = top; click a grid cell to change position)</span>
        </h3>
        <LineChart
          yLabel="log10(rank+1)"
          yMax={logMax}
          invertY
          xLabels={layers.map((l) => `L${l}`)}
          series={tracked.map((t, i) => ({
            name: t.token_display,
            color: COLORS[i % COLORS.length],
            values: t.ranks.map((row) => Math.log10(Math.max(0, row[col]) + 1)),
          }))}
        />
        <div style={{ marginTop: 6 }}>
          {tracked.map((t, i) => (
            <span key={t.token_id} className="pill" style={{ borderColor: COLORS[i % COLORS.length] }}>
              <span style={{ color: COLORS[i % COLORS.length] }}>■</span> {t.token_display} (id{" "}
              {t.token_id})
            </span>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>Lens score across layers — position {positions[col]}</h3>
        <LineChart
          yLabel="lens score"
          yMax={Math.max(1, ...tracked.flatMap((t) => t.scores.map((row) => row[col] ?? 0)))}
          xLabels={layers.map((l) => `L${l}`)}
          series={tracked.map((t, i) => ({
            name: t.token_display,
            color: COLORS[i % COLORS.length],
            values: t.scores.map((row) => Math.max(0, row[col] ?? 0)),
          }))}
        />
      </div>

      <div className="panel">
        <h3>
          Rank heatmap — layer × position for{" "}
          <select
            value={heatToken.token_id}
            onChange={(e) => setHeatmapToken(parseInt(e.target.value, 10))}
            style={{
              background: "var(--panel2)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              borderRadius: 4,
            }}
          >
            {tracked.map((t) => (
              <option key={t.token_id} value={t.token_id}>
                {t.token_display}
              </option>
            ))}
          </select>
        </h3>
        <div className="gridwrap" style={{ maxHeight: 360 }}>
          <table className="grid">
            <thead>
              <tr>
                <th>layer \ pos</th>
                {positions.map((p) => (
                  <th key={p}>{p}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...layers]
                .slice()
                .reverse()
                .map((layer) => {
                  const rowIdx = layers.indexOf(layer);
                  return (
                    <tr key={layer}>
                      <th>L{layer}</th>
                      {positions.map((p, colIdx) => {
                        const rank = heatToken.ranks[rowIdx]?.[colIdx] ?? -1;
                        const intensity =
                          rank < 0 ? 0 : 1 - Math.log10(rank + 1) / logMax;
                        return (
                          <td
                            key={p}
                            className="cell"
                            title={`rank ${rank}`}
                            style={{
                              background: `rgba(89, 201, 165, ${(0.05 + 0.6 * Math.max(0, intensity)).toFixed(3)})`,
                            }}
                            onClick={() => onSelect({ layer, position: p })}
                          >
                            {rank >= 0 && rank < 100 ? rank : rank >= 0 ? "·" : ""}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
        <div className="muted" style={{ marginTop: 6 }}>
          Brighter = higher J-lens rank (closer to top of vocabulary). Numbers shown when rank &lt;
          100.
        </div>
      </div>
    </div>
  );
}
