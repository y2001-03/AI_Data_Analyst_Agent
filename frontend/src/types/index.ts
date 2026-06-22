export interface ChartCard {
  title: string;
  type: string;
  description: string;
}

export interface DatasetColumnProfile {
  name: string;
  data_type: string;
  missing_count: number;
  unique_values: number;
}

export interface AnalysisTask {
  task_name: string;
  reasoning: string;
  expected_output: string;
}

export interface AIAnalysisResult {
  summary: string;
  suggestions: string[];
  tasks: AnalysisTask[];
}

export interface ExecutionResult {
  task_name: string;
  type: "stats" | "groupby" | "trend" | "chart" | "sql";
  data: {
    rows: Record<string, unknown>[];
    chart_type?: "bar" | "line";
    labels?: Array<string | number>;
    values?: Array<string | number>;
    status?: string;
    reason?: string;
  };
  chart: {
    chart_type: "bar" | "line";
    x: string[];
    y: number[];
  } | null;
}

export interface TraceEntry {
  node: string;
  status: string;
  input_summary: Record<string, unknown>;
  output_summary: Record<string, unknown>;
}

export interface DebugInfo {
  execution_path: string[];
  node_status: Record<string, string>;
  trace_log: TraceEntry[];
  fallback_summary: {
    understand: boolean;
    plan: boolean;
  };
}

export interface DatasetFileInfo {
  file_name: string;
  file_type: string;
  row_count: number;
  column_count: number;
  preview: Record<string, string | number | boolean | null>[];
  columns: DatasetColumnProfile[];
}

export interface DatasetUploadResponse {
  dataset_id?: string | null;
  file_info: DatasetFileInfo;
  ai_analysis: AIAnalysisResult;
  execution_results: ExecutionResult[];
  debug: DebugInfo;
}

export interface AnalysisResponse {
  session_id: string;
  answer: string;
  steps: string[];
  charts: ChartCard[];
  report_markdown: string | null;
}
