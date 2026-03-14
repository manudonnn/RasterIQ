"""
InsightVisualizer — Subagent that turns deep analysis data points into
rich, targeted Plotly visualizations.

Takes the DeepAnalysisResult from DeepAnalysisAgent and auto-generates
drill-down charts: bar, treemap, heatmap, waterfall, comparison matrices.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import duckdb
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.deep_analysis_agent import DeepAnalysisResult, InvestigationStep


@dataclass
class InsightChart:
    """A single visualization produced by the InsightVisualizer."""
    figure: Any               # plotly Figure
    title: str
    description: str
    importance: int = 1       # 1-5 importance rating


VISUALIZER_SYSTEM_PROMPT = """You are a data visualization expert. Given investigation steps and data,
you generate SQL queries to retrieve data suitable for visualization.

IMPORTANT RULES:
1. Respond with valid JSON only — no markdown, no code fences.
2. Every SQL query must use "df1" as the table name.
3. Only use columns from the schema. Do NOT invent columns.
4. Each visualization should reveal a distinct insight.
5. Limit results to reasonable sizes (max 20 rows per chart).

Available columns:
  ID, RO_ID, SRC_SYS, ORG_NM, CNT_STATE, LOB, RUN_NO, FILE_STATUS_CD,
  LATEST_STAGE_NM, PRE_PROCESSING_DURATION, MAPPING_APROVAL_DURATION,
  ISF_GEN_DURATION, DART_GEN_DURATION, DART_REVIEW_DURATION,
  DART_UI_VALIDATION_DURATION, SPS_LOAD_DURATION,
  AVG_PRE_PROCESSING_DURATION, AVG_MAPPING_APROVAL_DURATION,
  AVG_ISF_GEN_DURATION, AVG_DART_GEN_DURATION, AVG_DART_REVIEW_DURATION,
  AVG_DART_UI_VALIDATION_DURATION, AVG_SPS_LOAD_DURATION,
  PRE_PROCESSING_HEALTH, MAPPING_APROVAL_HEALTH, ISF_GEN_HEALTH,
  DART_GEN_HEALTH, DART_REVIEW_HEALTH, DART_UI_VALIDATION_HEALTH,
  SPS_LOAD_HEALTH, IS_STUCK, IS_FAILED, FAILURE_STATUS
"""


class InsightVisualizer:
    """
    Takes deep analysis results and generates rich, contextual charts.
    Uses LLM to decide the best visualizations, then builds them with Plotly.
    """

    MAX_CHARTS = 4  # Maximum charts to generate per analysis

    def __init__(self):
        self._init_error = None
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv(
                "OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free"
            )
            if not api_key:
                self._init_error = "OPENROUTER_API_KEY not found in environment"
                self.llm = None
            else:
                self.llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=model_name,
                    temperature=0,
                    max_retries=3,
                    request_timeout=30,
                )
        except Exception as e:
            self._init_error = str(e)
            self.llm = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def visualize(
        self,
        deep_result: DeepAnalysisResult,
        df1: pd.DataFrame,
        user_query: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[InsightChart]:
        """
        Generate targeted visualizations from deep analysis results.

        Returns a list of InsightChart objects ready to render in Streamlit.
        """
        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        charts: List[InsightChart] = []

        # Strategy 1: Build charts from key_data_points (structured data — always works)
        if deep_result.key_data_points:
            _progress(f"📊 Building {len(set(d['category'] for d in deep_result.key_data_points))} charts from investigation data...")
            charts.extend(self._charts_from_data_points(deep_result.key_data_points))

        # Strategy 2: LLM-generated charts — SKIPPED to avoid rate limits
        # (Uncomment if you have paid OpenRouter credits)
        # if self.llm and deep_result.investigation_steps:
        #     llm_charts = self._llm_driven_charts(deep_result, df1, user_query)
        #     charts.extend(llm_charts)

        # Strategy 3: Auto-generate from investigation step SQLs (always works)
        if len(charts) < 3 and deep_result.investigation_steps:
            _progress("📈 Auto-generating charts from investigation queries...")
            auto_charts = self._auto_charts_from_steps(deep_result.investigation_steps, df1)
            charts.extend(auto_charts)

        _progress(f"✅ Visualization complete — {len(charts[:self.MAX_CHARTS])} chart(s) ready")
        # Deduplicate and limit
        return charts[: self.MAX_CHARTS]

    # ------------------------------------------------------------------ #
    #  Strategy 1: Charts from structured data points                     #
    # ------------------------------------------------------------------ #
    def _charts_from_data_points(self, data_points: List[Dict]) -> List[InsightChart]:
        """Build charts from the key_data_points in DeepAnalysisResult."""
        charts = []
        if not data_points:
            return charts

        # Group data points by category
        by_category: Dict[str, List[Dict]] = {}
        for dp in data_points:
            cat = dp.get("category", "general")
            by_category.setdefault(cat, []).append(dp)

        for category, points in by_category.items():
            hint = points[0].get("chart_hint", "bar")
            labels = [p.get("label", "") for p in points]
            values = []
            for p in points:
                v = p.get("value", 0)
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = 0
                values.append(v)

            if not labels or not values:
                continue

            chart_df = pd.DataFrame({"Label": labels, "Value": values})

            y_title = points[0].get("y_label", "Value")
            chart_labels = {"Value": y_title, "Label": category.replace('_', ' ').title()}

            try:
                if hint == "pie":
                    fig = px.pie(chart_df, names="Label", values="Value",
                                 title=f"Deep Analysis — {category.replace('_', ' ').title()}",
                                 labels=chart_labels)
                elif hint == "treemap":
                    chart_df["Parent"] = category.replace('_', ' ').title()
                    fig = px.treemap(chart_df, path=["Parent", "Label"], values="Value",
                                     title=f"Deep Analysis — {category.replace('_', ' ').title()}",
                                     labels=chart_labels)
                elif hint == "line":
                    fig = px.line(chart_df, x="Label", y="Value",
                                  title=f"Deep Analysis — {category.replace('_', ' ').title()}", markers=True,
                                  labels=chart_labels)
                else:  # bar (default)
                    fig = px.bar(chart_df, x="Label", y="Value",
                                  title=f"Deep Analysis — {category.replace('_', ' ').title()}",
                                  labels=chart_labels)

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                charts.append(InsightChart(
                    figure=fig,
                    title=f"{category.title()} Breakdown",
                    description=f"Visualizing {len(points)} data points in {category}",
                    importance=3,
                ))
            except Exception:
                continue

        return charts

    # ------------------------------------------------------------------ #
    #  Strategy 2: LLM-driven chart generation                           #
    # ------------------------------------------------------------------ #
    def _llm_driven_charts(
        self,
        deep_result: DeepAnalysisResult,
        df1: pd.DataFrame,
        user_query: str,
    ) -> List[InsightChart]:
        """Ask LLM to generate SQL queries for insightful visualizations."""
        charts = []

        steps_summary = "\n".join(
            f"Step {s.iteration}: {s.question} → {s.insight_gained}"
            for s in deep_result.investigation_steps
        )

        prompt = f"""
USER QUERY: {user_query}

DEEP ANALYSIS NARRATIVE (summary):
{deep_result.narrative[:1500]}

INVESTIGATION STEPS:
{steps_summary}

Generate up to 3 targeted SQL queries for visualizations that would help the user
understand the deep analysis findings. Each query should produce a small result set
(max 15 rows) with clear labels suitable for charting.

Respond with JSON:
{{
  "charts": [
    {{
      "title": "Chart Title",
      "description": "What this reveals",
      "sql": "SELECT ... FROM df1 ... LIMIT 15",
      "chart_type": "bar|line|pie|heatmap|treemap",
      "x_column": "column_name_for_x_axis",
      "y_column": "column_name_for_y_axis",
      "color_column": "optional_color_column_or_null",
      "importance": 3
    }}
  ]
}}
"""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=VISUALIZER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            parsed = self._parse_json(resp.content)
            if not parsed or "charts" not in parsed:
                return charts

            for chart_spec in parsed["charts"][:3]:
                fig = self._build_chart_from_spec(chart_spec, df1)
                if fig:
                    charts.append(fig)
        except Exception:
            pass

        return charts

    def _build_chart_from_spec(self, spec: Dict, df1: pd.DataFrame) -> Optional[InsightChart]:
        """Execute SQL and build a Plotly chart from an LLM specification."""
        sql = spec.get("sql", "")
        if not sql or not sql.strip().upper().startswith("SELECT"):
            return None

        try:
            result_df = duckdb.query(sql).to_df()
        except Exception:
            return None

        if result_df.empty:
            return None

        chart_type = spec.get("chart_type", "bar")
        x_col = spec.get("x_column", result_df.columns[0])
        y_col = spec.get("y_column", result_df.columns[-1] if len(result_df.columns) > 1 else result_df.columns[0])
        color_col = spec.get("color_column")
        title = spec.get("title", "Deep Analysis Visualization")

        # Validate columns exist
        if x_col not in result_df.columns:
            x_col = result_df.columns[0]
        if y_col not in result_df.columns:
            y_col = result_df.columns[-1]
        if color_col and color_col not in result_df.columns:
            color_col = None

        try:
            if chart_type == "pie":
                fig = px.pie(result_df, names=x_col, values=y_col, title=title)
            elif chart_type == "line":
                fig = px.line(result_df, x=x_col, y=y_col, color=color_col,
                              title=title, markers=True)
            elif chart_type == "heatmap":
                # For heatmap, try pivoting if possible
                if len(result_df.columns) >= 3:
                    try:
                        pivot = result_df.pivot_table(
                            index=result_df.columns[0],
                            columns=result_df.columns[1],
                            values=result_df.columns[2],
                            aggfunc="sum",
                        ).fillna(0)
                        fig = px.imshow(pivot, title=title, color_continuous_scale="RdYlGn_r",
                                        aspect="auto")
                    except Exception:
                        fig = px.bar(result_df, x=x_col, y=y_col, title=title)
                else:
                    fig = px.bar(result_df, x=x_col, y=y_col, title=title)
            elif chart_type == "treemap":
                path_cols = [c for c in result_df.columns if result_df[c].dtype == "object"][:3]
                val_cols = [c for c in result_df.columns if result_df[c].dtype in ["int64", "float64"]]
                if path_cols and val_cols:
                    fig = px.treemap(result_df, path=path_cols, values=val_cols[0], title=title)
                else:
                    fig = px.bar(result_df, x=x_col, y=y_col, title=title)
            else:  # bar (default)
                fig = px.bar(result_df, x=x_col, y=y_col, color=color_col,
                             title=title, color_continuous_scale="Viridis")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            return InsightChart(
                figure=fig,
                title=title,
                description=spec.get("description", ""),
                importance=spec.get("importance", 2),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Strategy 3: Auto-generate charts from investigation SQL results    #
    # ------------------------------------------------------------------ #
    def _auto_charts_from_steps(
        self,
        steps: List[InvestigationStep],
        df1: pd.DataFrame,
    ) -> List[InsightChart]:
        """Re-run investigation queries and auto-chart the results."""
        charts = []
        for step in steps[:2]:  # Only first 2 steps
            sql = step.sql_query
            if not sql or not sql.strip().upper().startswith("SELECT"):
                continue
            try:
                result_df = duckdb.query(sql).to_df()
                if result_df.empty or len(result_df.columns) < 2:
                    continue

                x_col = result_df.columns[0]
                y_col = result_df.columns[1]

                # Pick chart type based on data shape
                if result_df[x_col].nunique() <= 6:
                    fig = px.pie(result_df, names=x_col, values=y_col,
                                 title=f"Investigation: {step.question[:60]}...")
                else:
                    fig = px.bar(result_df, x=x_col, y=y_col,
                                 title=f"Investigation: {step.question[:60]}...",
                                 color=y_col, color_continuous_scale="Turbo")

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                charts.append(InsightChart(
                    figure=fig,
                    title=step.question[:80],
                    description=step.insight_gained,
                    importance=2,
                ))
            except Exception:
                continue

        return charts

    # ------------------------------------------------------------------ #
    #  Utility                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Robustly parse JSON from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
            return None
