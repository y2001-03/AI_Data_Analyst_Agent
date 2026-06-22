<template>
  <div class="page">
    <header class="hero">
      <div>
        <h1>AI Data Analyst Agent</h1>
        <p>Upload data, ask questions, inspect charts, and review reports.</p>
      </div>
    </header>
    <main class="grid">
      <FileUploadPanel @uploaded="handleUpload" />
      <ChatPanel :dataset-id="datasetId" @analyzed="handleAnalysis" />
      <ChartPanel :result="analysisResult" />
      <ReportPanel :result="analysisResult" />
      <DebugPanel :debug="debugInfo" class="debug-span" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

import ChartPanel from "../components/ChartPanel.vue";
import ChatPanel from "../components/ChatPanel.vue";
import DebugPanel from "../components/DebugPanel.vue";
import FileUploadPanel from "../components/FileUploadPanel.vue";
import ReportPanel from "../components/ReportPanel.vue";
import type { DatasetUploadResponse, DebugInfo } from "../types";

const debugInfo = ref<DebugInfo | null>(null);
const datasetId = ref<string | null>(null);
const analysisResult = ref<DatasetUploadResponse | null>(null);

function handleUpload(payload: {
  file: File | null;
  result: DatasetUploadResponse | null;
}): void {
  datasetId.value = payload.result?.dataset_id ?? null;
  analysisResult.value = payload.result;
  debugInfo.value = payload.result?.debug ?? null;
}

function handleAnalysis(payload: DatasetUploadResponse | null): void {
  analysisResult.value = payload;
  debugInfo.value = payload?.debug ?? null;
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  padding: 24px;
  background:
    radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 35%),
    linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

.hero {
  margin-bottom: 24px;
}

.hero h1 {
  margin: 0 0 8px;
  font-size: 36px;
}

.hero p {
  margin: 0;
  color: #475569;
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.debug-span {
  grid-column: 1 / -1;
}

@media (max-width: 960px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
