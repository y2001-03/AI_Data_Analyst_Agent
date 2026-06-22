# 架构说明

## 总览

AI Data Analyst Agent 采用分层工作流架构，用于支持基于文件上传和对话提问的数据分析场景。前端负责上传文件和提交用户问题，后端使用 FastAPI 提供接口，并将编排职责交给 LangGraph 工作流。整个工作流负责完成数据集理解、分析任务规划、工具调度执行，以及结构化结果返回，包括图表、报告和调试信息。

## Mermaid 架构图

```mermaid
flowchart TD
    user[User / Analyst] --> fe[Frontend<br/>Vue 3 + TypeScript]
    fe --> api[FastAPI Backend]

    api --> upload[Upload API Entry]
    api --> chat[Chat API Entry]

    upload --> graph[LangGraph Workflow]
    chat --> graph

    graph --> understand[Understand Node]
    understand --> dataset_ctx[Dataset Context Service]
    understand --> memory[Memory Service]

    understand --> plan[Plan Node<br/>LLM Planner]
    plan --> memory
    plan --> dataset_ctx

    plan --> execute[Execute Node]
    execute --> registry[Tool Registry]

    registry --> stats[Stats Tool]
    registry --> groupby[GroupBy Tool]
    registry --> trend[Trend Tool]
    registry --> sql[SQL Tool<br/>DuckDB]

    execute --> results[Execution Results]
    results --> charts[Charts]
    results --> report[Report]
    results --> debug[Debug]

    memory -.context.-> graph
    dataset_ctx -.dataset metadata.-> graph

    plan -.fallback route.-> execute
    execute -.skip on tool failure.-> debug
```

## 上传流程

上传流程从前端上传面板开始，最终进入 FastAPI 的数据集上传接口。后端读取上传的 CSV 或 XLSX 文件，提取文件基础信息、字段结构、预览数据以及数据集上下文，并将这些内容作为 LangGraph 工作流的初始状态。随后工作流依次进入理解、规划和执行阶段，最终返回一个统一的结构化响应，其中包含文件信息、AI 分析结果、执行结果以及调试追踪数据。

## 对话流程

对话流程复用与上传流程一致的工作流模型，但入口从文件上传接口切换为对话接口。用户问题会与当前激活的数据集上下文以及可选的记忆上下文一起传入图工作流。图工作流负责判断问题意图、规划分析任务，并选择对应工具生成结果。这种设计保证了上传分析和后续追问共享同一套编排模型，便于保持上下文一致性。

## 规划流程

规划流程始于理解阶段输出数据集摘要和推荐分析方向之后。LLM Planner 会把这些理解结果转换为显式的分析任务，每个任务都带有任务名称、推理说明和预期输出。当规划器失败或超时时，系统会自动退回到 mock 或基于规则的任务生成方式，使整个工作流继续执行，而不会因为规划器不可用而中断请求。因此，规划层是一个具备容错能力的编排层，而不是单点依赖。

## 工具流程

执行节点通过 Tool Registry 模式统一管理分析工具，使工具调度更加明确且便于扩展。规划得到的任务会被映射到不同的专用工具，例如 Stats Tool、GroupBy Tool、Trend Tool，以及基于 DuckDB 的 SQL Tool。每个工具都返回结构化结果，后续可以被渲染为表格、图表、报告或调试信息。这样的分层设计让规划层专注于意图理解，而执行层专注于确定性的数据计算。

## 记忆流程

Memory Service 作为图工作流的旁路支撑服务存在，它提供上下文信息，而不是直接控制执行逻辑。它可以存储最近的用户意图、分析偏好、历史结果以及会话级提示。在后续请求中，图工作流可以读取这些上下文，以提升规划质量并支持跨轮连续分析。在当前架构中，记忆层被设计为支持长期分析对话的辅助服务。

## 调试流程

调试流程主要用于系统可观测性和面试展示。LangGraph 工作流在执行过程中，会将每个节点的执行路径、节点状态以及追踪日志写入图状态中。对于 fallback 分支和 skipped 步骤，系统也会显式记录，使前端 Debug Panel 可以高亮展示这些情况。这让开发者和面试官能够清楚看到系统尝试了什么、在哪一步发生了降级，以及最终结果是如何生成的。
