import pandas as pd
from typing import Dict, Any, List
import re

from .entity_profiler import EntityProfiler
from .semantic_clusterer import SemanticClusterer
from .graph_analyzer import GraphAnalyzer
from .timeline_analyzer import TimelineAnalyzer
from .anomaly_detector import AnomalyDetector
from .correlation_analyzer import CorrelationAnalyzer
from .root_cause_chainer import RootCauseChainer
from .retry_analyzer import RetryAnalyzer

from .scope import AnalysisScope

class AnalysisEngine:
    """
    Orchestrates all 8 analysis modules, collects their outputs, and manages
    passing insights to the LLM agent for interpretation.
    """
    
    def __init__(self):
        self.modules = {
            "entity_profiler": EntityProfiler(),
            "semantic_clusterer": SemanticClusterer(),
            "graph_analyzer": GraphAnalyzer(),
            "timeline_analyzer": TimelineAnalyzer(),
            "anomaly_detector": AnomalyDetector(),
            "correlation_analyzer": CorrelationAnalyzer(),
            "root_cause_chainer": RootCauseChainer(),
            "retry_analyzer": RetryAnalyzer()
        }

    def _parse_scope(self, query: str) -> AnalysisScope:
        """
        Simple heuristic logic to parse a user query into an AnalysisScope object.
        Ideally an LLM does this, but regex handles standard use-cases quickly.
        """
        scope = AnalysisScope()
        q = query.lower()
        
        # State parsing
        state_list = ["ks", "kansas", "ca", "california", "tx", "texas", "ny", "new york"]
        for st in state_list:
            if st in q:
                if st in ["ks", "kansas"]: scope.states.append("KS")
                if st in ["ca", "california"]: scope.states.append("CA")
                if st in ["tx", "texas"]: scope.states.append("TX")
                if st in ["ny", "new york"]: scope.states.append("NY")
                
        # LOB parsing
        if "medicare" in q: scope.lobs.append("Medicare HMO")
        if "commercial" in q: scope.lobs.append("Commercial PPO/EPO")
        if "medicaid" in q: scope.lobs.append("Medicaid FFS")
        
        # Deduplicate
        scope.states = list(set(scope.states))
        scope.lobs = list(set(scope.lobs))
        
        return scope

    def run_single(self, module_name: str, query: str, df1: pd.DataFrame, df2: pd.DataFrame, messages: list = None, progress_callback=None) -> Any:
        # Import later to prevent circular imports if necessary
        from .combiner import MultiSelectCombiner
        scope = self._parse_scope(query)
        combiner = MultiSelectCombiner(self, df1, df2, [module_name], scope)
        return combiner.run(user_query=query, messages=messages, progress_callback=progress_callback)

    def run_auto(self, query: str, df1: pd.DataFrame, df2: pd.DataFrame, messages: list = None, progress_callback=None) -> Any:
        from .combiner import MultiSelectCombiner
        scope = self._parse_scope(query)
        
        # Mock LLM Planner - pick 3 modules dynamically based on keywords
        selected = []
        q = query.lower()
        
        if "fail" in q or "worst" in q or "org" in q:
            selected.append("entity_profiler")
        if "root" in q or "why" in q or "trace" in q or "cause" in q:
            selected.append("root_cause_chainer")
        if "time" in q or "trend" in q or "month" in q:
            selected.append("timeline_analyzer")
        if "cluster" in q or "meaning" in q or "reason" in q:
            selected.append("semantic_clusterer")
        if "anomaly" in q or "weird" in q or "outlier" in q:
            selected.append("anomaly_detector")
        if "retry" in q or "fix" in q:
            selected.append("retry_analyzer")
            
        if not selected:
            # Default fallback analysis
            selected = ["entity_profiler", "root_cause_chainer", "timeline_analyzer"]
            
        combiner = MultiSelectCombiner(self, df1, df2, selected, scope)
        return combiner.run(user_query=query, messages=messages, progress_callback=progress_callback)
