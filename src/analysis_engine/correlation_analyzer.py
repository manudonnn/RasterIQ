import pandas as pd
import numpy as np
import plotly.express as px
from .base_module import BaseAnalysisModule

class CorrelationAnalyzer(BaseAnalysisModule):
    """
    Module 6 — Correlation Analysis
    What it does: Finds mathematical relationships between variables.
    Computes: Pearson correlation matrix across numeric columns.
    Output: Correlation heatmap + top insights plain English explanation.
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        severity = 1
        
        # Select numeric columns
        numeric_df = df1.select_dtypes(include=[np.number])
        
        # Drop ID columns or sparse columns that distract from real insight
        cols_to_drop = [c for c in numeric_df.columns if 'ID' in c.upper() or numeric_df[c].nunique() < 2]
        numeric_df = numeric_df.drop(columns=cols_to_drop)
        
        if numeric_df.shape[1] < 2:
            return {"findings": "Not enough numeric columns to compute correlations.", "chart": None, "severity": 1}
            
        # Compute Pearson correlation matrix
        corr_matrix = numeric_df.corr(method='pearson')
        
        # Find strongest correlations (excluding self-correlation of 1.0)
        # Unstack and filter
        corr_pairs = corr_matrix.unstack().reset_index()
        corr_pairs.columns = ['var1', 'var2', 'correlation']
        corr_pairs = corr_pairs[corr_pairs['var1'] != corr_pairs['var2']]
        
        # Drop duplicates (A-B is same as B-A)
        corr_pairs['abs_corr'] = corr_pairs['correlation'].abs()
        corr_pairs = corr_pairs.sort_values('abs_corr', ascending=False)
        corr_pairs = corr_pairs[~corr_pairs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]
        
        # Get top 5 correlations
        top_corrs = corr_pairs.head(5)
        
        findings = f"Analyzed {numeric_df.shape[1]} numeric variables for correlations. "
        
        if len(top_corrs) > 0:
            top_v1 = top_corrs.iloc[0]['var1']
            top_v2 = top_corrs.iloc[0]['var2']
            top_val = top_corrs.iloc[0]['correlation']
            
            direction = "positive" if top_val > 0 else "negative"
            findings += f"The strongest relationship is a **{direction} correlation ({top_val:.2f})** between **'{top_v1}'** and **'{top_v2}'**. "
            
            if abs(top_val) > 0.7:
                findings += f"Changes in '{top_v1}' are a clear leading indicator for '{top_v2}' outcomes. "
                severity = 3
            
            findings += f"\n\n→ **Recommended action:** Use '{top_v1}' as an early-warning metric to predict and preemptively resolve downstream '{top_v2}' failures."
        
        # Generate Heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation Matrix of Numeric Features"
        )
        
        return {
            "findings": findings,
            "chart": fig,
            "severity": severity
        }
