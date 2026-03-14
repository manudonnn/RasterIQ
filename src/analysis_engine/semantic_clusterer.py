import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
import plotly.express as px
from .base_module import BaseAnalysisModule

class SemanticClusterer(BaseAnalysisModule):
    """
    Module 2 — Semantic Clustering
    What it does: Groups FAILURE_STATUS free-text reasons by meaning using NLP.
    How it works: Extracts unique errors, embeds via sentence-transformers, clusters using K-Means.
    Output: A scatter plot where each dot is a failure reason, colored by cluster.
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        if 'FAILURE_STATUS' not in df1.columns:
            return {"findings": "FAILURE_STATUS column not found.", "chart": None, "severity": 1}
            
        # Drop nan and extract non-empty unique errors
        failures_series = df1['FAILURE_STATUS'].dropna().astype(str)
        failures_series = failures_series[failures_series.str.strip() != '']
        failures = failures_series.unique().tolist()
        
        if len(failures) < 3:
            return {
                "findings": "Not enough diverse failure reasons to cluster.",
                "chart": None,
                "severity": 1
            }
            
        # 1. Embed error messages
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(failures)
        
        # 2. Cluster using K-Means
        # Dynamically determine k based on the number of unique failures
        k = max(2, min(5, len(failures) // 2))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # 3. Reduce dimensionality to 2D for visualization
        # UMAP requires n_neighbors <= number of samples
        n_neighbors = min(15, len(failures) - 1)
        # Add safeguard for UMAP to not fail on extremely small datasets
        if n_neighbors < 2:
            return {
                "findings": "Not enough unique failure reasons for semantic clustering (need at least 3).",
                "chart": None,
                "severity": 1
            }
            
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # 4. Format plot data
        plot_df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'reason': failures,
            'cluster': [f"Cluster {c+1}" for c in clusters]
        })
        
        # Map frequencies for dot size
        failure_counts = failures_series.value_counts().to_dict()
        plot_df['count'] = plot_df['reason'].map(failure_counts)
        plot_df['size'] = plot_df['count'].apply(lambda x: max(10, min(50, x * 5))) 
        
        # Build Plotly Chart
        fig = px.scatter(
            plot_df, x='x', y='y', color='cluster',
            size='size', hover_name='reason', hover_data=['count'],
            title="Semantic Clusters of Failure Reasons"
        )
        
        fig.update_layout(xaxis_visible=False, yaxis_visible=False)
        
        # 5. Determine dominant cluster and build findings
        cluster_counts = plot_df.groupby('cluster')['count'].sum()
        top_cluster = cluster_counts.idxmax()
        top_count = cluster_counts.max()
        top_failures = plot_df[plot_df['cluster'] == top_cluster].sort_values('count', ascending=False)['reason'].tolist()
        examples = ", ".join([f"'{f}'" for f in top_failures[:2]])
        
        severity = 3
        findings = f"Grouped {len(failures)} unique failure reasons into {k} linguistic clusters. "
        findings += f"The dominant theme ({top_cluster}) accounts for {top_count} total failures, "
        findings += f"with errors like {examples}."
        
        findings += f"\n\n→ **Recommended action:** Review the documentation for the '{top_cluster}' pattern as it is the most frequent cross-stage semantic failure point."
        
        return {
            "findings": findings,
            "chart": fig,
            "severity": severity
        }
