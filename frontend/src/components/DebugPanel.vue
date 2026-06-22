<template>
  <el-card class="panel">
    <template #header>Debug Mode</template>
    <div v-if="!debug" class="empty">Execution trace will appear here after upload.</div>
    <template v-else>
      <div class="path">
        <div
          v-for="(node, index) in debug.execution_path"
          :key="`${node}-${index}`"
          class="path-item"
        >
          <el-tag :type="tagType(node)" effect="dark">{{ node }}</el-tag>
          <span v-if="index < debug.execution_path.length - 1" class="arrow">→</span>
        </div>
      </div>
      <div class="summary">
        <el-alert
          :title="`Understand fallback: ${debug.fallback_summary.understand ? 'yes' : 'no'}`"
          :type="debug.fallback_summary.understand ? 'warning' : 'success'"
          :closable="false"
          show-icon
        />
        <el-alert
          :title="`Plan fallback: ${debug.fallback_summary.plan ? 'yes' : 'no'}`"
          :type="debug.fallback_summary.plan ? 'warning' : 'success'"
          :closable="false"
          show-icon
        />
      </div>
      <el-collapse class="trace-list">
        <el-collapse-item
          v-for="(entry, index) in debug.trace_log"
          :key="`${entry.node}-${index}`"
          :name="String(index)"
        >
          <template #title>
            <div class="trace-title">
              <span>{{ entry.node }}</span>
              <el-tag :type="tagType(entry.node)" effect="plain">{{ entry.status }}</el-tag>
            </div>
          </template>
          <div class="trace-section">
            <div class="trace-label">Input</div>
            <pre>{{ stringify(entry.input_summary) }}</pre>
          </div>
          <div class="trace-section">
            <div class="trace-label">Output</div>
            <pre>{{ stringify(entry.output_summary) }}</pre>
          </div>
        </el-collapse-item>
      </el-collapse>
    </template>
  </el-card>
</template>

<script setup lang="ts">
import type { DebugInfo } from "../types";

const props = defineProps<{
  debug: DebugInfo | null;
}>();

function tagType(node: string): "success" | "warning" | "danger" | "info" {
  if (!props.debug) {
    return "info";
  }
  const status = props.debug.node_status[node];
  if (status === "success") {
    return "success";
  }
  if (status === "fallback") {
    return "danger";
  }
  if (status === "failed" || status === "skipped") {
    return "warning";
  }
  return "info";
}

function stringify(value: Record<string, unknown>): string {
  return JSON.stringify(value, null, 2);
}
</script>

<style scoped>
.panel {
  min-height: 320px;
}

.empty {
  color: #6b7280;
}

.path {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.path-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.arrow {
  color: #64748b;
}

.summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.trace-list {
  margin-top: 16px;
}

.trace-title {
  display: flex;
  align-items: center;
  gap: 12px;
}

.trace-section + .trace-section {
  margin-top: 12px;
}

.trace-label {
  margin-bottom: 6px;
  font-size: 12px;
  color: #475569;
  text-transform: uppercase;
}

pre {
  margin: 0;
  padding: 12px;
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 6px;
  overflow: auto;
  font-size: 12px;
}

@media (max-width: 960px) {
  .summary {
    grid-template-columns: 1fr;
  }
}
</style>
