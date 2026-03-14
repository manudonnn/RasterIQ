import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from src.analysis_engine.scope import AnalysisScope
from src.analysis_engine.engine import AnalysisEngine
from src.agent.deep_analysis_agent import DeepAnalysisAgent, DeepAnalysisResult, InvestigationStep
from src.agent.insight_visualizer import InsightVisualizer, InsightChart
from langchain_openai import ChatOpenAI
import plotly.express as px
import os

@dataclass
class DashboardLayout:
    columns: int
    charts: list
    titles: list

@dataclass
class CombinedResult:
    modules_run: list[str]
    per_module: dict[str, dict]
    synthesis: str
    layout: DashboardLayout
    choropleth_chart: Any = None
    sunburst_chart: Any = None
    # ── Deep Analysis fields ──
    deep_narrative: str = ""
    investigation_log: list = field(default_factory=list)
    deep_charts: list = field(default_factory=list)
    analysis_confidence: int = 0

class MultiSelectCombiner:
    def __init__(self, engine_instance: AnalysisEngine, df1: pd.DataFrame, df2: pd.DataFrame, modules: list[str], scope: AnalysisScope):
        self.engine = engine_instance
        self.df1 = df1
        self.df2 = df2
        self.modules = modules
        self.scope = scope
        self.results = {}
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                model=model_name,
                max_retries=3,
                request_timeout=30,
            )
        except Exception:
            self.llm = None

        # Initialize subagents
        self.deep_agent = DeepAnalysisAgent()
        self.visualizer = InsightVisualizer()

    def run(self, user_query: str = "", messages: list = None, progress_callback=None) -> CombinedResult:
        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # Step 1 — run all selected modules in parallel
        _progress(f"⚡ Running {len(self.modules)} analysis modules in parallel...")
        futures = {}
        with ThreadPoolExecutor() as pool:
            for name in self.modules:
                if name in self.engine.modules:
                    futures[name] = pool.submit(self.engine.modules[name].run, self.df1, self.df2, self.scope)
            
            for name, future in futures.items():
                res = future.result()
                res['title'] = name.replace('_', ' ').title()
                res['key_insight'] = res.get('findings', '')
                self.results[name] = res
                _progress(f"✅ Module complete: {name.replace('_', ' ').title()}")

        # Step 2 — LLM synthesizes cross-module insight
        _progress("🧩 Synthesizing cross-module insights...")
        synthesis = self._synthesize()

        # Step 3 — build unified dashboard layout
        _progress("📐 Building dashboard layout...")
        layout = self._build_dashboard_layout()
        
        # Build extra synthetic visuals if >= 3 modules
        choro, sunb = None, None
        if len(self.modules) >= 3:
            choro = self._build_choropleth()
            sunb = self._build_sunburst()

        # ─────────────────────────────────────────────────────────────
        # Step 4 — Deep Analysis Subagent
        # ─────────────────────────────────────────────────────────────
        _progress("🔬 Starting Deep Analysis subagent...")
        module_findings = {
            name: res.get("findings", "") for name, res in self.results.items()
        }
        query_for_deep = user_query if user_query else "Analyze the roster processing pipeline failures."

        deep_result: DeepAnalysisResult = self.deep_agent.analyze(
            user_query=query_for_deep,
            module_findings=module_findings,
            df1=self.df1,
            df2=self.df2,
            messages=messages,
            progress_callback=progress_callback,
        )

        # ─────────────────────────────────────────────────────────────
        # Step 5 — Insight Visualizer Subagent
        # ─────────────────────────────────────────────────────────────
        _progress("🎨 Starting Insight Visualizer subagent...")
        insight_charts: List[InsightChart] = self.visualizer.visualize(
            deep_result=deep_result,
            df1=self.df1,
            user_query=query_for_deep,
            progress_callback=progress_callback,
        )

        return CombinedResult(
            modules_run=self.modules,
            per_module=self.results,
            synthesis=synthesis,
            layout=layout,
            choropleth_chart=choro,
            sunburst_chart=sunb,
            deep_narrative=deep_result.narrative,
            investigation_log=[
                {
                    "step": s.iteration,
                    "question": s.question,
                    "sql": s.sql_query,
                    "result_preview": s.query_result[:300],
                    "insight": s.insight_gained,
                }
                for s in deep_result.investigation_steps
            ],
            deep_charts=[ic.figure for ic in insight_charts],
            analysis_confidence=deep_result.confidence,
        )

    def _synthesize(self) -> str:
        findings = "\n".join([
            f"{name}: {res.get('key_insight', '')}"
            for name, res in self.results.items()
        ])
        
        prompt = f"""
        You ran {len(self.modules)} analysis modules on a roster dataset with the following scope: {self.scope}.
        Here are the findings from each:
        {findings}
        
        Write a 3-sentence executive summary that:
        1. States the single most critical finding
        2. Connects findings across modules (e.g. entity profiling + root cause agree on X)
        3. Recommends one immediate action.
        """
        
        if self.llm:
            try:
                msg = self.llm.invoke(prompt)
                return msg.content
            except Exception as e:
                error_msg = str(e)
                return f"❌ **LLM Synthesis Failed**\n\n**Error:** `{error_msg}`\n\n**Raw Findings:**\n{findings}"
        else:
            return f"⚠️ **LLM Not Initialized** — check your `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in `.env`\n\n**Raw Findings:**\n{findings}"

    def _build_dashboard_layout(self) -> DashboardLayout:
        n = len(self.modules)
        cols = 2 if n >= 3 else 1
        return DashboardLayout(
            columns=cols,
            charts=[self.results[m].get('chart') for m in self.modules],
            titles=[self.results[m].get('title') for m in self.modules]
        )
        
    def _build_choropleth(self):
        filtered_df = self.engine.modules[self.modules[0]]._apply_scope(self.df1, self.scope)
        if 'CNT_STATE' in filtered_df.columns and not filtered_df.empty:
            state_scores = filtered_df.groupby('CNT_STATE')['IS_FAILED'].mean().reset_index()
            # Lower failure = higher health
            state_scores['Health Score'] = (1 - state_scores['IS_FAILED']) * 100
            
            fig = px.choropleth(
                state_scores,
                locations='CNT_STATE',
                locationmode="USA-states",
                color='Health Score',
                scope="usa",
                color_continuous_scale="RdYlGn",
                title="Cross-Module Combined Health Score by State"
            )
            return fig
        return None
        
    def _build_sunburst(self):
        filtered_df = self.engine.modules[self.modules[0]]._apply_scope(self.df1, self.scope)
        req = ['CNT_STATE', 'LOB', 'ORG_NM', 'LATEST_STAGE_NM']
        if all(c in filtered_df.columns for c in req) and not filtered_df.empty:
            b_df = filtered_df[filtered_df['IS_FAILED'] == 1].dropna(subset=req)
            if not b_df.empty:
                fig = px.sunburst(
                    b_df,
                    path=['CNT_STATE', 'LOB', 'ORG_NM', 'LATEST_STAGE_NM'],
                    title="Volume Drill-Down (State ➔ LOB ➔ Org ➔ Stage)"
                )
                return fig
        return None
