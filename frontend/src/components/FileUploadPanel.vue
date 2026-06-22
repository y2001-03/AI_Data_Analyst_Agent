<template>
  <el-card class="panel">
    <template #header>File Upload</template>
    <el-upload
      drag
      action="#"
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFileChange"
      accept=".csv,.xlsx"
    >
      <div>Drop CSV or XLSX here</div>
      <div class="hint">The backend will detect schema and preview rows.</div>
    </el-upload>
    <el-alert
      v-if="errorMessage"
      class="feedback"
      :title="errorMessage"
      type="error"
      show-icon
      :closable="false"
    />
    <el-skeleton v-if="isLoading" :rows="4" animated class="feedback" />
    <div v-if="result" class="result">
      <div class="meta">
        <span>{{ result.file_info.file_name }}</span>
        <span>{{ result.file_info.row_count }} rows</span>
        <span>{{ result.file_info.column_count }} columns</span>
      </div>
      <el-table :data="result.file_info.columns" size="small">
        <el-table-column prop="name" label="Column" />
        <el-table-column prop="data_type" label="Type" />
        <el-table-column prop="missing_count" label="Missing" />
        <el-table-column prop="unique_values" label="Unique" />
      </el-table>
      <el-table :data="result.file_info.preview" size="small" class="preview-table">
        <el-table-column
          v-for="column in previewColumns"
          :key="column"
          :prop="column"
          :label="column"
          min-width="120"
        />
      </el-table>
      <div class="summary-block">
        <h4>AI Summary</h4>
        <p>{{ result.ai_analysis.summary }}</p>
      </div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { ElMessage } from "element-plus";
import type { UploadFile, UploadFiles } from "element-plus";

import { uploadDataset } from "../api/files";
import type { DatasetUploadResponse } from "../types";

const isLoading = ref(false);
const errorMessage = ref("");
const result = ref<DatasetUploadResponse | null>(null);
const emit = defineEmits<{
  uploaded: [payload: { file: File | null; result: DatasetUploadResponse | null }];
}>();

const previewColumns = computed(() => {
  const firstRow = result.value?.file_info.preview[0];
  return firstRow ? Object.keys(firstRow) : [];
});
const sessionStorageKey = "ai-data-analyst-session-id";

async function handleFileChange(uploadFile: UploadFile, _files: UploadFiles): Promise<void> {
  if (!uploadFile.raw) {
    return;
  }
  isLoading.value = true;
  errorMessage.value = "";
  try {
    result.value = await uploadDataset(uploadFile.raw, getSessionId());
    emit("uploaded", { file: uploadFile.raw, result: result.value });
    ElMessage.success("File uploaded successfully.");
  } catch (error) {
    errorMessage.value = "File upload failed. Please check the file format.";
    result.value = null;
    emit("uploaded", { file: null, result: null });
  } finally {
    isLoading.value = false;
  }
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
  min-height: 220px;
}

.hint {
  margin-top: 8px;
  color: #64748b;
  font-size: 13px;
}

.feedback {
  margin-top: 16px;
}

.result {
  margin-top: 16px;
}

.meta {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
  color: #475569;
  flex-wrap: wrap;
}

.preview-table {
  margin-top: 16px;
}

.summary-block {
  margin-top: 16px;
}

.summary-block h4 {
  margin: 0 0 8px;
  font-size: 14px;
}

.summary-block p {
  margin: 0;
  color: #334155;
  line-height: 1.5;
}
</style>
