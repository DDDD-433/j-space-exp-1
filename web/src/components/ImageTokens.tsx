import { useMemo, useState } from "react";
import type { PositionInfo, RunResult, SelectedCell } from "../types";

interface Props {
  run: RunResult;
  selected: SelectedCell | null;
  onSelect: (cell: SelectedCell) => void;
}

export function ImageTokens({ run, selected, onSelect }: Props) {
  const [imageIndex, setImageIndex] = useState(0);

  const imagePositions = useMemo(
    () => run.positions.filter((p) => p.modality === "image_token"),
    [run],
  );
  const imageIndices = useMemo(
    () => [...new Set(imagePositions.map((p) => p.image_index ?? 0))].sort((a, b) => a - b),
    [imagePositions],
  );
  const layers = run.metadata.layers.filter(
    (l) => !run.cells.find((c) => c.layer === l)?.is_model_output,
  );
  const readLayer = selected?.layer ?? layers[Math.floor(layers.length / 2)] ?? 0;

  if (imagePositions.length === 0) {
    return (
      <div className="empty">
        No image-token positions in this run. Upload an image and use a VLM to populate this tab.
      </div>
    );
  }

  const mapping = run.metadata.patch_mapping;
  const current = imagePositions.filter((p) => (p.image_index ?? 0) === imageIndices[imageIndex]);
  const imageInfo = run.images.find((im) => im.image_index === imageIndices[imageIndex]);
  const gridRows = imageInfo?.grid_rows ?? null;
  const gridCols = imageInfo?.grid_cols ?? null;
  const hasGeometry =
    mapping !== "unavailable" &&
    gridRows !== null &&
    gridCols !== null &&
    current.every((p) => p.patch_row !== null && p.patch_col !== null);

  const conceptAt = (position: number): string => {
    const cell = run.cells.find((c) => c.layer === readLayer && c.position === position);
    return cell?.jlens_top[0]?.token_display ?? "";
  };

  return (
    <div>
      <div className="muted" style={{ marginBottom: 8 }}>
        Patch mapping: <b>{mapping}</b>.{" "}
        {mapping === "approximate" &&
          "Positions correspond to merged/shuffled patch groups, not single vision-encoder patches."}
        {mapping === "unavailable" &&
          "This architecture resamples visual tokens; no spatial coordinates exist, so a token strip is shown instead."}{" "}
        Readouts at image positions answer: which text concepts is this visual-token activation
        disposed to make the decoder verbalize? They do not decode the raw vision encoder.
      </div>
      <div style={{ marginBottom: 8, display: "flex", gap: 12, alignItems: "center" }}>
        {imageIndices.length > 1 && (
          <span>
            Image{" "}
            <select
              value={imageIndex}
              onChange={(e) => setImageIndex(parseInt(e.target.value, 10))}
              style={{
                background: "var(--panel2)",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: 4,
              }}
            >
              {imageIndices.map((idx, i) => (
                <option key={idx} value={i}>
                  #{idx}
                </option>
              ))}
            </select>
          </span>
        )}
        <span className="muted">
          Reading layer L{readLayer} (select a cell in the grid to change layer)
        </span>
      </div>

      {hasGeometry && imageInfo && imageInfo.data_uri ? (
        <div className="panel">
          <h3>
            Patch overlay — {gridRows}×{gridCols} decoder-token grid ({mapping})
          </h3>
          <div className="imgwrap">
            <img src={imageInfo.data_uri} alt="input" />
            {current.map((p: PositionInfo) => (
              <div
                key={p.index}
                className={`patch ${selected?.position === p.index ? "selected" : ""}`}
                title={`pos ${p.index} · patch (${p.patch_row},${p.patch_col}) · top: ${conceptAt(p.index)}`}
                style={{
                  left: `${((p.patch_col ?? 0) / (gridCols ?? 1)) * 100}%`,
                  top: `${((p.patch_row ?? 0) / (gridRows ?? 1)) * 100}%`,
                  width: `${100 / (gridCols ?? 1)}%`,
                  height: `${100 / (gridRows ?? 1)}%`,
                }}
                onClick={() => onSelect({ layer: readLayer, position: p.index })}
              />
            ))}
          </div>
          <div className="muted" style={{ marginTop: 6 }}>
            Click a patch to select its decoder position, then explore it across layers in the
            other tabs.
          </div>
        </div>
      ) : (
        <div className="panel">
          <h3>Image-token strip (no spatial mapping{mapping === "approximate" ? " geometry recorded" : ""})</h3>
          <div className="strip">
            {current.map((p) => (
              <span
                key={p.index}
                className={`chip ${selected?.position === p.index ? "selected" : ""}`}
                title={`top concept at L${readLayer}: ${conceptAt(p.index)}`}
                onClick={() => onSelect({ layer: readLayer, position: p.index })}
              >
                {p.index}
              </span>
            ))}
          </div>
        </div>
      )}

      {selected && (
        <div className="panel">
          <h3>Top concepts across layers — position {selected.position}</h3>
          <table className="detail">
            <thead>
              <tr>
                <th>layer</th>
                <th>top J-lens concepts</th>
              </tr>
            </thead>
            <tbody>
              {[...layers].reverse().map((layer) => {
                const cell = run.cells.find(
                  (c) => c.layer === layer && c.position === selected.position,
                );
                return (
                  <tr key={layer}>
                    <td>L{layer}</td>
                    <td>
                      {cell?.jlens_top.slice(0, 6).map((t, i) => (
                        <span key={i} className="tok" style={{ marginRight: 6 }}>
                          {t.token_display}
                        </span>
                      ))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
