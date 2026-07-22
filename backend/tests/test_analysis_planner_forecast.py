"""Planner and workflow tests for forecast intent boundaries."""

from app.graph.dataset_workflow import DatasetGraphNodes
from app.schemas.file import AnalysisTask, DatasetColumnProfile, DatasetUploadResponse
from app.services.analysis_planner_service import AnalysisPlannerService


def _info() -> DatasetUploadResponse:
    return DatasetUploadResponse(file_name="sales.csv", file_type="csv", row_count=100, column_count=4,
        preview=[], columns=[
            DatasetColumnProfile(name="date", data_type="datetime64[ns]", missing_count=0, unique_values=100),
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=0, unique_values=90),
            DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=4),
            DatasetColumnProfile(name="price", data_type="float64", missing_count=0, unique_values=80)])


def test_planner_recognizes_future_horizon_columns_and_metric() -> None:
    task = AnalysisPlannerService()._mock_tasks(_info(), "预测未来 14 天 sales，计算 RMSE")[0]
    assert task.type == "forecast_analysis"
    assert task.params["horizon"] == 14 and task.params["target_column"] == "sales"
    assert task.params["time_column"] == "date" and task.params["primary_metric"] == "rmse"


def test_planner_parses_next_week_and_models() -> None:
    planner = AnalysisPlannerService()
    moving = planner._mock_tasks(_info(), "使用移动平均预测 sales 下周会是多少，窗口 5")[0]
    linear = planner._mock_tasks(_info(), "使用线性趋势预测未来 sales")[0]
    seasonal = planner._mock_tasks(_info(), "使用季节性 naive 预测未来 sales，周期 7")[0]
    assert moving.params["horizon"] == 7 and moving.params["method"] == "moving_average" and moving.params["window"] == 5
    assert linear.params["method"] == "linear_trend"
    assert seasonal.params["method"] == "seasonal_naive" and seasonal.params["season_length"] == 7


def test_planner_parses_grouped_forecast() -> None:
    task = AnalysisPlannerService()._mock_tasks(_info(), "按地区分别预测未来 sales")[0]
    assert task.type == "forecast_analysis" and task.params["mode"] == "grouped"
    assert task.params["group_by"] == "region" and task.params["target_column"] == "sales"


def test_historical_trend_and_existing_routes_are_preserved() -> None:
    planner = AnalysisPlannerService()
    assert planner._mock_tasks(_info(), "历史 sales 趋势怎么样")[0].type == "trend"
    assert planner._mock_tasks(_info(), "检查 sales 异常值")[0].type == "anomaly_detection"
    assert planner._mock_tasks(_info(), "分析 sales 分布和偏度")[0].type == "distribution_analysis"
    assert planner._mock_tasks(_info(), "分析 sales 和 price 相关性")[0].type == "correlation_analysis"
    tasks = planner._mock_tasks(_info(), "按地区求 sales 总额")
    assert any(task.type == "groupby" for task in tasks)


def test_future_word_without_prediction_intent_is_not_forecast() -> None:
    task = AnalysisPlannerService()._mock_tasks(_info(), "未来数据字段如何命名")[0]
    assert task.type != "forecast_analysis"


def test_normalized_forecast_defaults_and_prompt_boundary() -> None:
    planner = AnalysisPlannerService()
    task = planner._normalize_task({"task_name": "Forecast", "type": "forecast_analysis",
                                    "reason": "Predict future values.", "params": {}}, _info())
    assert task.params == {"mode": "univariate", "method": "auto", "horizon": 7, "primary_metric": "mae"}
    assert "historical patterns do not guarantee" in task.expected_output
    prompt = planner._system_prompt()
    assert "forecast_analysis" in prompt and "not historical trend questions" in prompt


def test_workflow_does_not_append_visualization_to_forecast() -> None:
    task = AnalysisTask(task_name="Forecast", reasoning="Predict future sales.", expected_output="Forecast chart.",
                        type="forecast_analysis", params={})
    assert DatasetGraphNodes()._ensure_visualization_task(_info(), [task]) == [task]
