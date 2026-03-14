import pandas as pd
import duckdb
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from .base_module import BaseAnalysisModule

class RootCauseChainer(BaseAnalysisModule):
    """
    Module 7 — Root Cause Chaining
    What it does: Traces market issues down to the specific org, LOB, source, or stage.
    How it works: Chained SQL queries traversing failure dimensions.
    Output: A causal chain diagram (PyVis) + generated narrative report.
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        severity = 1
        
        # We need specific columns to trace causation
        req_cols = ['CNT_STATE', 'ORG_NM', 'LATEST_STAGE_NM', 'IS_FAILED']
        if not all(col in df1.columns for col in req_cols):
            return {"findings": "Missing required columns for Root Cause Chaining.", "chart": None, "severity": 1}
            
        # 1. Find the worst market (State)
        market_query = '''
        SELECT CNT_STATE, SUM(IS_FAILED) as fails, COUNT(*) as total
        FROM df1
        GROUP BY CNT_STATE
        ORDER BY fails DESC
        LIMIT 1
        '''
        try:
            worst_market_df = duckdb.query(market_query).to_df()
            if worst_market_df.empty or worst_market_df['fails'][0] == 0:
                return {"findings": "No failures found to chain.", "chart": None, "severity": 1}
        except Exception as e:
            return {"findings": f"Query failed: {str(e)}", "chart": None, "severity": 1}
            
        worst_market = worst_market_df['CNT_STATE'][0]
        market_fails = worst_market_df['fails'][0]
        
        # 2. Find the worst org in that market
        org_query = f'''
        SELECT ORG_NM, SUM(IS_FAILED) as fails
        FROM df1
        WHERE CNT_STATE = '{worst_market}' AND IS_FAILED = 1
        GROUP BY ORG_NM
        ORDER BY fails DESC
        LIMIT 1
        '''
        worst_org_df = duckdb.query(org_query).to_df()
        worst_org = worst_org_df['ORG_NM'][0]
        org_fails = worst_org_df['fails'][0]
        
        # 3. Find the exact stage failing for that org
        stage_query = f'''
        SELECT LATEST_STAGE_NM, COUNT(*) as fails
        FROM df1
        WHERE ORG_NM = '{worst_org}' AND IS_FAILED = 1
        GROUP BY LATEST_STAGE_NM
        ORDER BY fails DESC
        LIMIT 1
        '''
        worst_stage_df = duckdb.query(stage_query).to_df()
        worst_stage = worst_stage_df['LATEST_STAGE_NM'][0]
        stage_fails = worst_stage_df['fails'][0]
        
        # Construct Narrative
        findings = f"Root Cause Analysis discovered the highest risk market is **'{worst_market}'** ({market_fails} failures). "
        findings += f"This is being driven by **'{worst_org}'** ({org_fails} market failures). "
        findings += f"The pipeline bottleneck for this provider occurs specifically at the **'{worst_stage}'** stage "
        findings += f"({stage_fails} of their {org_fails} failures happen exactly here)."
        
        findings += f"\n\n→ **Recommended action:** Expedite a manual review of the '{worst_stage}' queue for market '{worst_market}'. Check for recent schema changes in the '{worst_org}' source files."
        
        if org_fails > 10: severity = 4
        if org_fails > 50: severity = 5
        
        # Construct Causal Chain Graph
        G = nx.DiGraph()
        G.add_edge("Pipeline Failures", worst_market, title=f"{market_fails} fails")
        G.add_edge(worst_market, worst_org, title=f"{org_fails} fails")
        G.add_edge(worst_org, worst_stage, title=f"Bottleneck: {stage_fails} fails")
        
        net = Network(height="300px", width="100%", bgcolor="white", font_color="black", directed=True)
        
        for node in G.nodes:
            color = "#ff4b4b" if node == worst_stage else "#1f77b4"
            net.add_node(node, label=str(node), color=color, shape="box")
            
        for edge in G.edges:
            net.add_edge(edge[0], edge[1])
            
        net.set_options("""
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed"
            }
          }
        }
        """)
        
        html_path = tempfile.mktemp(suffix=".html")
        net.write_html(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        if os.path.exists(html_path):
            os.remove(html_path)
            
        return {
            "findings": findings,
            "chart": None,
            "html": html_content,
            "severity": severity
        }
