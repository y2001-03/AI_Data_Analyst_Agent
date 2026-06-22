import { http } from "./http";
import type { DatasetUploadResponse } from "../types";

interface ChatAnalysisRequest {
  session_id?: string;
  question: string;
  dataset_id?: string;
}

export async function analyzeDatasetQuestion(
  payload: ChatAnalysisRequest,
): Promise<DatasetUploadResponse> {
  const response = await http.post<DatasetUploadResponse>("/v1/analysis/chat", payload);
  return response.data;
}
