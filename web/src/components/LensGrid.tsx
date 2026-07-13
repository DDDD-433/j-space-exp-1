import { useMemo, useState } from "react";
import type { CellRecord, PositionInfo, RunResult, SelectedCell } from "../types";

interface Props {
  run: RunResult;
  selected: SelectedCell | null;
  onSelect: (cell: SelectedCell) => void;
}

interface TooltipState {
  x: number;
  y: number;
  cell: CellRecord;
  info: PositionInfo | undefined;
}

function heat(score: number): string {
  const s = Math.max(0, Math.min(1, score));
  return `rgba(110, 168, 254, ${(0.06 + 0.5 * s).toFixed(3)})`;
}

export function LensGrid({ run, selected, onSelect }: Props) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const { layers, positions, cellMap, posInfo } = useMemo(() => {
    const cellMap = new Map<string, CellRecord>();
    for (const cell of run.cells) cellMap.set(`${cell.layer}:${cell.position}`, cell);
    const layers = [...new Set(run.cells.map((c) => c.layer))].sort((a, b) => b - a);
    const positions = [...new Set(run.cells.map((c) => c.position))].sort((a, b) => a - b);
    const posInfo = new Map(run.positions.map((p) => [p.index, p]));
    return { layers, positions, cellMap, posInfo };
  }, [run]);

  return (
    <>
      <div className="legend">
        <span>cell text: top J-lens concept · intensity: normalized lens score (display scale)</span>
        <span>
          <span className="swatch" style={{ borderBottom: "2px solid var(--green)" }} /> image token
        </span>
        <span>
          <span className="swatch" style={{ borderBottom: "2px dotted var(--purple)" }} /> special
        </span>
        <span>bottom row = model output distribution (J = I)</span>
      </div>
      <div className="gridwrap">
        <table className="grid">
          <thead>
            <tr>
              <th>layer \ pos</th>
              {positions.map((p) => {
                const info = posInfo.get(p);
                return (
                  <th key={p} title={info ? `${info.modality} · token ${info.token_id}` : ""}>
                    {p}
                    <br />
                    <span className="tok">{info?.token_text?.trim() || "·"}</span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {layers.map((layer) => {
              const first = cellMap.get(`${layer}:${positions[0]}`);
              const isOutput = first?.is_model_output ?? false;
              return (
                <tr key={layer} className={isOutput ? "output-row" : ""}>
                  <th>{isOutput ? `L${layer} (output)` : `L${layer}`}</th>
                  {positions.map((p) => {
                    const cell = cellMap.get(`${layer}:${p}`);
                    if (!cell) return <td key={p} />;
                    const top = cell.jlens_top[0];
                    const info = posInfo.get(p);
                    const isSelected =
                      selected?.layer === layer && selected?.position === p;
                    return (
                      <td
                        key={p}
                        className={`cell mod-${info?.modality ?? "unknown"}${isSelected ? " selected" : ""}`}
                        style={{ background: top ? heat(top.normalized_score) : undefined }}
                        onClick={() => onSelect({ layer, position: p })}
                        onMouseMove={(e) =>
                          setTooltip({ x: e.clientX, y: e.clientY, cell, info })
                        }
                        onMouseLeave={() => setTooltip(null)}
                      >
                        {top ? top.token_display.trim() || "·" : ""}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {tooltip && (
        <div
          className="tooltip"
          style={{
            left: Math.min(tooltip.x + 14, window.innerWidth - 380),
            top: Math.min(tooltip.y + 14, window.innerHeight - 300),
          }}
        >
          <b>
            L{tooltip.cell.layer} · pos {tooltip.cell.position}
          </b>
          {tooltip.info && (
            <>
              {" "}
              · {tooltip.info.modality} · source{" "}
              <span className="tok">{tooltip.info.token_text ?? ""}</span>
            </>
          )}
          <br />
          <span className="muted">
            ‖h‖ = {tooltip.cell.activation_norm.toFixed(1)} · ‖Jh‖ ={" "}
            {tooltip.cell.transported_norm.toFixed(1)}
          </span>
          <table className="detail">
            <thead>
              <tr>
                <th>#</th>
                <th>top J-lens concepts</th>
                <th>lens score</th>
              </tr>
            </thead>
            <tbody>
              {tooltip.cell.jlens_top.slice(0, 10).map((t, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    <span className="tok">{t.token_display}</span>
                  </td>
                  <td>{t.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
