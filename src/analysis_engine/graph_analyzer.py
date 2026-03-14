import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from .base_module import BaseAnalysisModule

class GraphAnalyzer(BaseAnalysisModule):
    """
    Module 3 — Graph Node Analysis
    What it does: Builds a network graph of orgs, states, source systems, and LOBs to identify central bottlenecks.
    How it works: ORG_NM → SRC_SYS → CNT_STATE → LOB. Runs PageRank/centrality.
    Output: An interactive network diagram (pyvis/networkx).
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        severity = 1
        
        # Build a graph: ORG_NM → SRC_SYS → CNT_STATE → LOB
        G = nx.Graph()
        
        # We'll extract edges from df1
        # Dropping NAs to ensure valid nodes
        req_cols = ['ORG_NM', 'SRC_SYS', 'CNT_STATE', 'LOB']
        if not all(col in df1.columns for col in req_cols):
            return {"findings": "Missing required columns for Graph Analysis.", "chart": None, "severity": 1}
            
        edges_df = df1[req_cols + (['IS_FAILED'] if 'IS_FAILED' in df1.columns else [])].dropna()
        
        # Track failures per node to highlight high-risk nodes (red)
        failure_counts = {}
        if 'IS_FAILED' in edges_df.columns:
            for _, row in edges_df.iterrows():
                if row['IS_FAILED'] == 1:
                    for col in req_cols:
                        failure_counts[row[col]] = failure_counts.get(row[col], 0) + 1
        
        # Add edges
        for _, row in edges_df.drop_duplicates(subset=['ORG_NM', 'SRC_SYS']).iterrows():
            G.add_edge(row['ORG_NM'], row['SRC_SYS'])
        for _, row in edges_df.drop_duplicates(subset=['SRC_SYS', 'CNT_STATE']).iterrows():
            G.add_edge(row['SRC_SYS'], row['CNT_STATE'])
        for _, row in edges_df.drop_duplicates(subset=['CNT_STATE', 'LOB']).iterrows():
            G.add_edge(row['CNT_STATE'], row['LOB'])
            
        if len(G.nodes) == 0:
            return {"findings": "No graph data could be constructed.", "chart": None, "severity": 1}
            
        # Run PageRank to find centrality/importance
        try:
            centrality = nx.pagerank(G)
        except Exception:
            # Fallback to degree centrality if PageRank fails to converge
            centrality = nx.degree_centrality(G)
            
        # Find the most critical node
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_node, top_score = sorted_nodes[0]
        
        findings = f"Generated a network of **{len(G.nodes)}** nodes and **{len(G.edges)}** edges. "
        findings += f"The most central risk node is **'{top_node}'**. "
        
        if top_node in failure_counts and failure_counts[top_node] > 0:
            findings += f"This node is tied to **{failure_counts[top_node]}** failures, meaning issues here can leverage a cascade effect. "
            severity = 4
        elif failure_counts:
            findings += "This central hub does not currently have tied failures, but remains a critical bottleneck. "
            
        findings += f"\n\n→ **Recommended action:** Isolate processing for high-risk entities connected to the '{top_node}' hub to prevent a system-wide failure cascade."
        
        # Build interactive PyVis Network
        net = Network(height="400px", width="100%", bgcolor="white", font_color="black")
        
        for node in G.nodes:
            # Scale node size based on centrality
            size = max(10, centrality.get(node, 0.01) * 300)
            
            # Highlight nodes with high failure associations in red
            fails = failure_counts.get(node, 0)
            color = "#ff4b4b" if fails > 0 else "#1f77b4"
            title = f"Node: {node}<br>Centrality: {centrality.get(node, 0):.3f}<br>Failures: {fails}"
            
            net.add_node(node, label=str(node), title=title, size=size, color=color)
            
        for edge in G.edges:
            net.add_edge(edge[0], edge[1])
            
        net.repulsion(node_distance=150, spring_length=200)
        
        # PyVis generates HTML, we can parse it and pack it into the result dictionary
        # the dashboard / app will render this raw HTML.
        html_path = tempfile.mktemp(suffix=".html")
        net.write_html(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        if os.path.exists(html_path):
            os.remove(html_path)
            
        return {
            "findings": findings,
            "chart": None,          # Standard plot
            "html": html_content,   # Pass custom PyVis interactive HTML
            "severity": severity
        }
