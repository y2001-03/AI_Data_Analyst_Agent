<template>
  <el-card class="panel">
    <template #header>Chat</template>

    <el-input
      v-model="question"
      :rows="6"
      type="textarea"
      placeholder="Ask a data question..."
    />

    <el-alert
      v-if="errorMessage"
      class="feedback"
      :title="errorMessage"
      type="error"
      show-icon
      :closable="false"
    />

    <el-button class="action" type="primary" :loading="isLoading" @click="onAnalyze">
      Analyze
    </el-button>

    <div v-if="answer" class="answer">
      <h4>AI Answer</h4>
      <p>{{ answer }}</p>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";

import { analyzeDatasetQuestion } from "../api/analysis";
import type { DatasetUploadResponse } from "../types";

const props = defineProps<{
  datasetId: string | null;
}>();

const emit = defineEmits<{
  analyzed: [payload: DatasetUploadResponse | null];
}>();

const question = ref("");
const answer = ref("");
const errorMessage = ref("");
const isLoading = ref(false);
const sessionStorageKey = "ai-data-analyst-session-id";

async function onAnalyze(): Promise<void> {
  if (!props.datasetId) {
    errorMessage.value = "Please upload a dataset before asking a question.";
    answer.value = "";
    emit("analyzed", null);
    return;
  }
  if (!question.value.trim()) {
    errorMessage.value = "Please enter a question for the analysis agent.";
    answer.value = "";
    emit("analyzed", null);
    return;
  }

  isLoading.value = true;
  errorMessage.value = "";

  try {
    const sessionId = getSessionId();
    const result = await analyzeDatasetQuestion({
      session_id: sessionId,
      question: question.value,
      dataset_id: props.datasetId,
    });
    answer.value = buildAnswer(question.value, result);
    emit("analyzed", result);
    ElMessage.success("Analysis completed.");
  } catch {
    errorMessage.value = "Analysis failed. Please try uploading the file again.";
    answer.value = "";
    emit("analyzed", null);
  } finally {
    isLoading.value = false;
  }
}

function buildAnswer(questionText: string, result: DatasetUploadResponse): string {
  const summary = result.ai_analysis.summary;
  const firstResult = result.execution_results[0];
  if (!firstResult) {
    return summary;
  }

  const firstLine = formatExecutionResult(firstResult);
  if (!questionText.trim()) {
    return `${summary} ${firstLine}`.trim();
  }

  return `Question: ${questionText}. ${summary} ${firstLine}`.trim();
}

function formatExecutionResult(result: DatasetUploadResponse["execution_results"][number]): string {
  if (!result.data.rows.length) {
    return result.data.reason ?? `${result.task_name} returned no rows.`;
  }
  return `${result.task_name}: ${JSON.stringify(result.data.rows.slice(0, 3))}`;
}

function getSessionId(): string {
  const existing = window.localStorage.getItem(sessionStorageKey);
  if (existing) {
    return existing;
  }
  const created = window.crypto.randomUUID();
  window.localStorage.setItem(sessionStorageKey, created);
  return created;
}
</script>

<style scoped>
.panel {
  min-height: 260px;
}

.feedback {
  margin-top: 16px;
}

.action {
  margin-top: 16px;
}

.answer {
  margin-top: 16px;
}

.answer h4 {
  margin: 0 0 8px;
  font-size: 14px;
}

.answer p {
  margin: 0;
  color: #334155;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
