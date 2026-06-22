<template>
  <el-card class="panel">
    <template #header>Charts</template>
    <div v-if="!chartConfigs.length" class="placeholder">Visualization results will appear here.</div>
    <div v-else class="chart-list">
      <div v-for="chart in chartConfigs" :key="chart.id" class="chart-card">
        <h4>{{ chart.title }}</h4>
        <div :ref="(element) => setChartRef(chart.id, element)" class="chart-canvas"></div>
      </div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, watch } from "vue";
import * as echarts from "echarts";
import type { ECharts, EChartsOption } from "echarts";

import type { DatasetUploadResponse, ExecutionResult } from "../types";

const props = defineProps<{
  result: DatasetUploadResponse | null;
}>();

interface ChartConfig {
  id: string;
  title: string;
  option: EChartsOption;
}

const chartElements = new Map<string, HTMLDivElement>();
const chartInstances = new Map<string, ECharts>();

const chartConfigs = computed<ChartConfig[]>(() => {
  const results = props.result?.execution_results ?? [];
  return results
    .map((result, index) => buildChartConfig(result, index))
    .filter((item): item is ChartConfig => item !== null);
});

watch(
  chartConfigs,
  async () => {
    await nextTick();
    renderCharts();
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  disposeCharts();
});

function setChartRef(id: string, element: unknown): void {
  if (element instanceof HTMLDivElement) {
    chartElements.set(id, element);
    return;
  }
  chartElements.delete(id);
}

function renderCharts(): void {
  const activeIds = new Set(chartConfigs.value.map((chart) => chart.id));
  for (const [id, instance] of chartInstances.entries()) {
    if (!activeIds.has(id)) {
      instance.dispose();
      chartInstances.delete(id);
    }
  }
  for (const chart of chartConfigs.value) {
    const element = chartElements.get(chart.id);
    if (!element) {
      continue;
    }
    const existing = chartInstances.get(chart.id);
    const instance = existing ?? echarts.init(element);
    instance.setOption(chart.option, true);
    chartInstances.set(chart.id, instance);
  }
}

function disposeCharts(): void {
  for (const instance of chartInstances.values()) {
    instance.dispose();
  }
  chartInstances.clear();
}

function buildChartConfig(result: ExecutionResult, index: number): ChartConfig | null {
  if (result.chart?.x.length && result.chart.y.length) {
    return {
      id: `${result.task_name}-${index}`,
      title: result.task_name,
      option: {
        tooltip: { trigger: "axis" },
        grid: { left: 36, right: 20, top: 40, bottom: 32, containLabel: true },
        xAxis: {
          type: "category",
          data: result.chart.x,
        },
        yAxis: {
          type: "value",
        },
        series: [
          {
            name: result.task_name,
            type: result.chart.chart_type === "line" ? "line" : "bar",
            smooth: result.chart.chart_type === "line",
            data: result.chart.y,
          },
        ],
      },
    };
  }
  const records = result.data.rows ?? [];
  if (!records.length) {
    return null;
  }
  const firstRow = records[0];
  const keys = Object.keys(firstRow);
  if (keys.length < 2) {
    return null;
  }
  const xKey = keys[0];
  const seriesKeys = keys.slice(1).filter((key) =>
    records.some((row) => typeof row[key] === "number"),
  );
  if (!seriesKeys.length) {
    return null;
  }
  return {
    id: `${result.task_name}-${index}`,
    title: result.task_name,
    option: {
      tooltip: { trigger: "axis" },
      legend: { data: seriesKeys },
      grid: { left: 36, right: 20, top: 40, bottom: 32, containLabel: true },
      xAxis: {
        type: "category",
        data: records.map((row) => String(row[xKey] ?? "")),
      },
      yAxis: {
        type: "value",
      },
      series: seriesKeys.map((key) => ({
        name: key,
        type: result.type === "trend" ? "line" : "bar",
        smooth: result.type === "trend",
        data: records.map((row) => toNumber(row[key])),
      })),
    },
  };
}

function toNumber(value: unknown): number {
  return typeof value === "number" ? value : Number(value ?? 0);
}
</script>

<style scoped>
.panel {
  min-height: 260px;
}

.placeholder {
  color: #6b7280;
}

.chart-list {
  display: grid;
  gap: 16px;
}

.chart-card h4 {
  margin: 0 0 8px;
  font-size: 14px;
}

.chart-canvas {
  width: 100%;
  height: 280px;
}
</style>
