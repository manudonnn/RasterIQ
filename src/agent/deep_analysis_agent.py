"""
DeepAnalysisAgent — Iterative deep analysis subagent.

PRIMARY PATH: Pure DuckDB/pandas analytics engine — runs 6 targeted
investigation queries against the roster data to extract deep insights
without any LLM dependency, producing structured findings + visualization data.

LLM ENHANCEMENT: When the LLM is available (not rate-limited), it adds a
narrative layer on top of the hard analytics. If it fails, the analytics
still produce full output.

This ensures the deep analysis ALWAYS works regardless of LLM availability.
"""

import os
import json
import time
import pandas as pd
import duckdb
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class InvestigationStep:
    """One iteration of the deep analysis loop."""
    iteration: int
    question: str
    sql_query: str
    query_result: str
    insight_gained: str


@dataclass
class DeepAnalysisResult:
    """Output of the DeepAnalysisAgent."""
    narrative: str
    investigation_steps: List[InvestigationStep] = field(default_factory=list)
    key_data_points: List[Dict[str, Any]] = field(default_factory=list)
    confidence: int = 1
    iterations_run: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# The 6 core investigation queries — always run regardless of LLM availability
# ──────────────────────────────────────────────────────────────────────────────
INVESTIGATION_QUERIES = [
    {
        "question": "Which organizations have the highest failure rates?",
        "sql": """
            SELECT ORG_NM,
                   COUNT(*) AS total,
                   SUM(IS_FAILED) AS failures,
                   ROUND(100.0 * SUM(IS_FAILED) / COUNT(*), 1) AS failure_pct,
                   MODE(FAILURE_STATUS) AS top_failure_reason
            FROM df1
            WHERE ORG_NM IS NOT NULL
            GROUP BY ORG_NM
            ORDER BY failure_pct DESC
            LIMIT 10
        """,
        "chart_hint": "bar",
        "category": "organization_failures",
        "x_col": "ORG_NM",
        "y_col": "failure_pct",
        "y_label": "Failure Rate (%)",
    },
    {
        "question": "Which pipeline stages are the biggest bottlenecks?",
        "sql": """
            SELECT LATEST_STAGE_NM,
                   COUNT(*) AS total,
                   SUM(IS_FAILED) AS failures,
                   ROUND(100.0 * SUM(IS_FAILED) / COUNT(*), 1) AS failure_pct,
                   AVG(RUN_NO) AS avg_retries
            FROM df1
            WHERE LATEST_STAGE_NM IS NOT NULL
            GROUP BY LATEST_STAGE_NM
            ORDER BY failures DESC
            LIMIT 10
        """,
        "chart_hint": "bar",
        "category": "stage_bottlenecks",
        "x_col": "LATEST_STAGE_NM",
        "y_col": "failure_pct",
        "y_label": "Failure Rate (%)",
    },
    {
        "question": "What are the most common failure reasons and their volume?",
        "sql": """
            SELECT FAILURE_STATUS,
                   COUNT(*) AS count,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_all
            FROM df1
            WHERE FAILURE_STATUS IS NOT NULL AND IS_FAILED = 1
            GROUP BY FAILURE_STATUS
            ORDER BY count DESC
            LIMIT 10
        """,
        "chart_hint": "pie",
        "category": "failure_reasons",
        "x_col": "FAILURE_STATUS",
        "y_col": "count",
        "y_label": "Number of Failed Records",
    },
    {
        "question": "Which states have the worst health scores and failure concentration?",
        "sql": """
            SELECT CNT_STATE,
                   COUNT(*) AS total,
                   SUM(IS_FAILED) AS failures,
                   SUM(IS_STUCK) AS stuck,
                   ROUND(100.0 * SUM(IS_FAILED) / COUNT(*), 1) AS failure_pct,
                   ROUND(AVG(RUN_NO), 2) AS avg_retries
            FROM df1
            WHERE CNT_STATE IS NOT NULL
            GROUP BY CNT_STATE
            ORDER BY failure_pct DESC
        """,
        "chart_hint": "bar",
        "category": "state_performance",
        "x_col": "CNT_STATE",
        "y_col": "failure_pct",
        "y_label": "State Failure Rate (%)",
    },
    {
        "question": "How do retry counts correlate with failure rates by LOB?",
        "sql": """
            SELECT LOB,
                   COUNT(*) AS total,
                   ROUND(AVG(RUN_NO), 2) AS avg_retries,
                   MAX(RUN_NO) AS max_retries,
                   SUM(IS_FAILED) AS failures,
                   ROUND(100.0 * SUM(IS_FAILED) / COUNT(*), 1) AS failure_pct
            FROM df1
            WHERE LOB IS NOT NULL
            GROUP BY LOB
            ORDER BY avg_retries DESC
        """,
        "chart_hint": "bar",
        "category": "lob_retry_analysis",
        "x_col": "LOB",
        "y_col": "avg_retries",
        "y_label": "Average Retry Count",
    },
    {
        "question": "What is the duration bottleneck — which stage takes longest on average?",
        "sql": """
            SELECT 
                'PRE_PROCESSING' AS stage, AVG(PRE_PROCESSING_DURATION) AS avg_duration, AVG(AVG_PRE_PROCESSING_DURATION) AS baseline
            FROM df1
            UNION ALL
            SELECT 'MAPPING_APPROVAL', AVG(MAPPING_APROVAL_DURATION), AVG(AVG_MAPPING_APROVAL_DURATION) FROM df1
            UNION ALL
            SELECT 'ISF_GEN', AVG(ISF_GEN_DURATION), AVG(AVG_ISF_GEN_DURATION) FROM df1
            UNION ALL
            SELECT 'DART_GEN', AVG(DART_GEN_DURATION), AVG(AVG_DART_GEN_DURATION) FROM df1
            UNION ALL
            SELECT 'DART_REVIEW', AVG(DART_REVIEW_DURATION), AVG(AVG_DART_REVIEW_DURATION) FROM df1
            UNION ALL
            SELECT 'SPS_LOAD', AVG(SPS_LOAD_DURATION), AVG(AVG_SPS_LOAD_DURATION) FROM df1
            ORDER BY avg_duration DESC
        """,
        "chart_hint": "bar",
        "category": "duration_analysis",
        "x_col": "stage",
        "y_col": "avg_duration",
        "y_label": "Avg Duration (Units)",
    },
    {
        "question": "How do stuck ROs (df1) correlate with market SCS percentage (df2)?",
        "sql": """
            SELECT df1.CNT_STATE as market,
                   SUM(df1.IS_STUCK) as stuck_ros,
                   ROUND(AVG(df2.SCS_PERCENT), 1) as avg_scs_percent
            FROM df1
            JOIN df2 ON df1.CNT_STATE = df2.MARKET
            WHERE df1.CNT_STATE IS NOT NULL
            GROUP BY df1.CNT_STATE
            ORDER BY stuck_ros DESC
        """,
        "chart_hint": "bar",
        "category": "cross_metric_stuck_scs",
        "x_col": "market",
        "y_col": "stuck_ros",
        "y_label": "Count of Stuck Rosters",
    },
]

SYSTEM_PROMPT = """You are a healthcare roster data analyst. Given investigation findings from SQL queries,
write a deep, structured narrative analysis. Be specific — cite exact numbers, organization names, and stage names.

IMPORTANT: Respond with valid JSON only. No markdown fences. No extra text."""


class DeepAnalysisAgent:
    """
    Iterative deep-analysis subagent.

    Always runs 6 targeted DuckDB analytics queries. LLM is used
    only for narrative synthesis — if rate-limited, falls back to
    a structured rule-based narrative derived from the query results.
    """

    MAX_ITERATIONS = 3

    def __init__(self):
        self._init_error = None
        self.llm = None
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
            if not api_key:
                self._init_error = "OPENROUTER_API_KEY not found in environment"
            else:
                self.llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=model_name,
                    temperature=0,
                    max_retries=2,
                    request_timeout=25,
                )
        except Exception as e:
            self._init_error = str(e)

    def analyze(
        self,
        user_query: str,
        module_findings: Dict[str, str],
        df1: pd.DataFrame,
        df2: pd.DataFrame = None,
        messages: list = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> DeepAnalysisResult:
        """
        Run the deep analysis. Always uses DuckDB analytics queries.
        Optionally enhances with LLM narrative if available.
        """
        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # ── Step 1: Run all investigation queries (no LLM needed) ──
        _progress("🔬 Running deep cross-table data investigation queries...")
        steps, data_points = self._run_investigation_queries(df1, df2, _progress)

        # ── Step 2: Build narrative ──
        if self.llm is not None:
            _progress("🧠 Requesting LLM narrative synthesis...")
            narrative = self._llm_narrative(user_query, module_findings, steps, messages, _progress)
        else:
            _progress("📊 Building rule-based narrative from data (LLM unavailable)...")
            narrative = None

        if not narrative:
            # Always works — builds narrative from the actual data
            narrative = self._rule_based_narrative(steps, module_findings, user_query)
            _progress("✅ Deep analysis complete (data-driven narrative)")
        else:
            _progress("✅ Deep analysis complete (LLM-enhanced narrative)")

        return DeepAnalysisResult(
            narrative=narrative,
            investigation_steps=steps,
            key_data_points=data_points,
            confidence=4 if self.llm else 3,
            iterations_run=len(steps),
        )

    # ────────────────────────────────────────────────────────── #
    #  Always-on: DuckDB investigation queries                  #
    # ────────────────────────────────────────────────────────── #
    def _run_investigation_queries(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        _progress: Callable,
    ):
        """Run all 6 investigation queries and return steps + data points."""
        steps: List[InvestigationStep] = []
        data_points: List[Dict[str, Any]] = []

        for i, q in enumerate(INVESTIGATION_QUERIES, 1):
            _progress(f"🔍 [{i}/{len(INVESTIGATION_QUERIES)}] {q['question']}")
            result_str, result_df = self._run_query(q["sql"], df1, df2)

            # Extract data points for visualization
            if result_df is not None and not result_df.empty:
                x_col = q["x_col"]
                y_col = q["y_col"]
                if x_col in result_df.columns and y_col in result_df.columns:
                    for _, row in result_df.head(8).iterrows():
                        try:
                            val = float(row[y_col])
                        except (ValueError, TypeError):
                            val = 0
                        data_points.append({
                            "label": str(row[x_col]),
                            "value": val,
                            "y_label": q.get("y_label", "Value"),
                            "category": q["category"],
                            "chart_hint": q["chart_hint"],
                        })

            # Build a concise insight from the raw data
            insight = self._data_insight(q["question"], result_df)

            steps.append(InvestigationStep(
                iteration=i,
                question=q["question"],
                sql_query=q["sql"].strip(),
                query_result=result_str[:400],
                insight_gained=insight,
            ))

        return steps, data_points

    def _run_query(self, sql: str, df1: pd.DataFrame, df2: pd.DataFrame = None):
        """Execute a DuckDB query, return (string, DataFrame)."""
        try:
            result_df = duckdb.query(sql.strip()).to_df()
            if len(result_df) > 20:
                result_df = result_df.head(20)
            return result_df.to_string(index=False), result_df
        except Exception as e:
            return f"QUERY ERROR: {str(e)}", None

    def _data_insight(self, question: str, df: Optional[pd.DataFrame]) -> str:
        """Derive a one-line insight directly from the query result DataFrame."""
        if df is None or df.empty:
            return "No data returned for this query."
        try:
            cols = df.columns.tolist()
            numeric_cols = [c for c in cols if df[c].dtype in ["float64", "int64"]]
            label_cols = [c for c in cols if df[c].dtype == "object"]

            if label_cols and numeric_cols:
                top_label = str(df[label_cols[0]].iloc[0])
                top_val = df[numeric_cols[0]].iloc[0]
                total_rows = len(df)
                return (
                    f"Top result: '{top_label}' with {numeric_cols[0]}={top_val:.1f}. "
                    f"({total_rows} total rows)"
                )
            return f"Result has {len(df)} rows and {len(cols)} columns."
        except Exception:
            return f"Result: {len(df)} rows."

    # ────────────────────────────────────────────────────────── #
    #  Rule-based narrative (no LLM required)                   #
    # ────────────────────────────────────────────────────────── #
    def _rule_based_narrative(
        self,
        steps: List[InvestigationStep],
        module_findings: Dict[str, str],
        user_query: str,
    ) -> str:
        """Build a structured narrative purely from data findings."""
        lines = [
            f"## 🔬 Deep Data Analysis\n",
            f"**Query:** _{user_query}_\n",
            f"**Method:** 6 targeted SQL investigations on the roster dataset\n",
            "---\n",
        ]

        for step in steps:
            lines.append(f"### {step.iteration}. {step.question}")
            lines.append(f"> {step.insight_gained}\n")

        lines.append("---\n")
        lines.append("### 📋 Module Context\n")
        for name, finding in module_findings.items():
            lines.append(f"**{name.replace('_', ' ').title()}:** {finding}\n")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────── #
    #  LLM enhancement (optional)                               #
    # ────────────────────────────────────────────────────────── #
    def _llm_narrative(
        self,
        user_query: str,
        module_findings: Dict[str, str],
        steps: List[InvestigationStep],
        messages: list,
        _progress: Callable,
    ) -> Optional[str]:
        """Attempt LLM synthesis. Returns None on failure."""
        steps_text = "\n".join(
            f"Query {s.iteration}: {s.question}\nInsight: {s.insight_gained}"
            for s in steps
        )
        module_text = "\n".join(f"[{k}] {v}" for k, v in module_findings.items())
        
        # Build conversational context
        ctx_str = ""
        if messages:
            # Get the last 3 user queries (excluding the current one we just appended)
            past_qs = [m["content"] for m in messages if m.get("role") == "user" and m.get("content") != user_query][-3:]
            if past_qs:
                ctx_str = "\nPAST CONVERSATION CONTEXT:\nThe user previously asked: " + " | ".join(past_qs) + "\n"

        prompt = f"""
USER QUERY: {user_query}
{ctx_str}
MODULE FINDINGS:
{module_text}

DEEP DATA INVESTIGATION RESULTS:
{steps_text}

Write a comprehensive 3-paragraph analysis that:
1. States the root cause with specific data evidence from the queries
2. Explains how the different factors interconnect
3. Gives 2-3 specific, actionable recommendations

Respond with JSON: {{"narrative": "...", "confidence": 4}}
"""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            parsed = self._parse_json(resp.content)
            if parsed and "narrative" in parsed:
                return parsed["narrative"]
            return None
        except Exception as e:
            _progress(f"⚠️ LLM narrative failed (using data-driven narrative): {str(e)[:80]}")
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
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

    def _fallback(self, module_findings: Dict[str, str]) -> DeepAnalysisResult:
        """Last-resort fallback."""
        text = "\n".join(f"**{k}:** {v}" for k, v in module_findings.items())
        return DeepAnalysisResult(
            narrative=f"## Raw Module Findings\n\n{text}",
            confidence=1,
        )
