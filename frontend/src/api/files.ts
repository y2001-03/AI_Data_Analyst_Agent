import { http } from "./http";
import type { DatasetUploadResponse } from "../types";

export async function uploadDataset(
  file: File,
  sessionId?: string,
): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionId) {
    formData.append("session_id", sessionId);
  }
  const response = await http.post<DatasetUploadResponse>("/v1/datasets/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
  return response.data;
}
