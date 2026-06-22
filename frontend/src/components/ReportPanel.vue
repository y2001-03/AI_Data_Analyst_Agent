<template>
  <el-card class="panel">
    <template #header>Report</template>
    <div v-if="!result" class="placeholder">Generated report content will appear here.</div>
    <div v-else class="report">
      <h4>AI Summary</h4>
      <p>{{ result.ai_analysis.summary }}</p>
      <h4>Execution Results</h4>
      <div
        v-for="item in result.execution_results"
        :key="item.task_name"
        class="result-block"
      >
        <strong>{{ item.task_name }}</strong>
        <pre>{{ formatResult(item.data) }}</pre>
      </div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import type { DatasetUploadResponse } from "../types";

defineProps<{
  result: DatasetUploadResponse | null;
}>();

function formatResult(data: DatasetUploadResponse["execution_results"][number]["data"]): string {
  return JSON.stringify(data, null, 2);
}
</script>

<style scoped>
.panel {
  min-height: 260px;
}

.placeholder {
  color: #6b7280;
}

.report h4 {
  margin: 0 0 8px;
  font-size: 14px;
}

.report p {
  margin: 0 0 16px;
  color: #334155;
  line-height: 1.6;
}

.result-block {
  margin-top: 12px;
}

.result-block pre {
  margin: 8px 0 0;
  padding: 12px;
  border-radius: 12px;
  background: #f8fafc;
  color: #1e293b;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
