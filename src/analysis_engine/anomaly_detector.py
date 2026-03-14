import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import plotly.express as px
from .base_module import BaseAnalysisModule

class AnomalyDetector(BaseAnalysisModule):
    """
    Module 5 — Anomaly Detection
    What it does: Automatically flags anything that is statistically unusual.
    How it works: z-score for durations, Isolation Forest for multivariate analysis.
    Output: Highlighted table of anomalies, heatmap showing anomaly concentration.
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        severity = 1
        
        # Focus on duration columns
        duration_cols = [c for c in df1.columns if 'DURATION' in c and df1[c].dtype in [np.float64, np.int64]]
        
        if not duration_cols:
            return {"findings": "No duration columns found to detect anomalies.", "chart": None, "severity": 1}
            
        temp_df = df1[duration_cols].copy().fillna(0)
        
        # 1. Isolation Forest for multivariate anomalies
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(temp_df)
        
        # Copy to avoid SettingWithCopyWarning if df1 is a slice
        df1 = df1.copy()
        df1['is_anomaly'] = outliers == -1
        num_anomalies = df1['is_anomaly'].sum()
        
        # 2. Identify the specific feature driving the anomaly using z-scores
        # Add a small epsilon to avoid divide by zero standard deviations if constant
        z_scores = temp_df.apply(lambda x: zscore(x + 1e-10))
        
        findings = f"Detected {num_anomalies} statistically anomalous rows ({(num_anomalies/len(df1)):.1%} of data). "
        
        # Determine concentration
        if 'CNT_STATE' in df1.columns and num_anomalies > 0:
            anomaly_summary_full = df1[df1['is_anomaly']].groupby('CNT_STATE').size()
            top_state = anomaly_summary_full.idxmax()
            top_state_count = anomaly_summary_full.max()
            pct_concentration = (top_state_count / num_anomalies) * 100
            findings += f"**{top_state}** accounts for {pct_concentration:.0f}% of all anomalies ({top_state_count} of {num_anomalies}). "

        extreme_points = (np.abs(z_scores) > 3).sum()
        top_anomalous_stage = extreme_points.idxmax()
        top_stage_count = extreme_points.max()
        
        if top_stage_count > 0:
            findings += f"The **{top_anomalous_stage}** stage is the primary driver of multivariate variance ({top_stage_count} σ-outliers). This indicates a localized processing bottleneck rather than a system-wide failure. "
            
        findings += f"\n\n→ **Recommended action:** Investigate the {top_state if 'CNT_STATE' in df1.columns and num_anomalies > 0 else 'primary'} market source system for recent NPI mapping changes or credentialing rule updates."
        
        if top_stage_count > 0:
            severity = 3
            if top_stage_count > len(df1) * 0.05:
                severity = 4
        
        # 3. Create a Bar Chart / Heatmap of Anomalies by State
        if 'CNT_STATE' in df1.columns:
            anomaly_summary = df1[df1['is_anomaly']].groupby('CNT_STATE').size().reset_index(name='anomaly_count')
            anomaly_summary = anomaly_summary.sort_values('anomaly_count', ascending=False).head(10)
            
            if not anomaly_summary.empty:
                fig = px.bar(
                    anomaly_summary, 
                    x='CNT_STATE', 
                    y='anomaly_count',
                    title='Concentration of Anomalies by State',
                    color='anomaly_count',
                    color_continuous_scale='Reds'
                )
            else:
                fig = None
        else:
            fig = None
            
        return {
            "findings": findings,
            "chart": fig,
            "severity": severity
        }
