import type {
  ArtifactInfo,
  DecompositionRecord,
  ModelInfo,
  RunResult,
  UploadInfo,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  loadModel: (model_id: string, device: string, dtype: string) =>
    post<ModelInfo>("/api/models/load", { model_id, device, dtype }),
  listArtifacts: () => request<ArtifactInfo[]>("/api/artifacts"),
  inspect: (body: Record<string, unknown>) => post<RunResult>("/api/inspect", body),
  decompose: (body: Record<string, unknown>) =>
    post<DecompositionRecord>("/api/decompose", body),
  upload: async (file: File): Promise<UploadInfo> => {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/uploads", { method: "POST", body: form });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(String(body.detail));
    }
    return response.json();
  },
};
