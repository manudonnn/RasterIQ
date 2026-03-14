import pandas as pd
import plotly.express as px
from .base_module import BaseAnalysisModule

class RetryAnalyzer(BaseAnalysisModule):
    """
    Module 8 — Retry Pattern Analysis
    What it does: Analyzes records w/ RUN_NO > 1 to determine if retrying fixes errors.
    Computes: Retry lift, latency added by retries, org-specific retry effectiveness.
    Output: Stacked bar chart showing first-pass vs retry recovery vs failure.
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        severity = 1
        
        req_cols = ['ORG_NM', 'RUN_NO', 'IS_FAILED']
        if not all(col in df1.columns for col in req_cols):
            return {"findings": "Missing required columns for Retry Analysis.", "chart": None, "severity": 1}
            
        # Group data to see how many succeeded on run 1, run > 1, or failed eventually
        df1_temp = df1.copy()
        
        # Categorize every record
        def categorize_retry(row):
            if row['IS_FAILED'] == 1:
                return 'Failed (Permanent)'
            if row['RUN_NO'] == 1:
                return 'Succeeded (First Try)'
            return 'Succeeded (After Retry)'
            
        df1_temp['Outcome'] = df1_temp.apply(categorize_retry, axis=1)
        
        retry_stats = df1_temp.groupby(['ORG_NM', 'Outcome']).size().reset_index(name='Count')
        
        # If no retries are happening, exit early gracefully
        if not (df1_temp['RUN_NO'] > 1).any():
            return {"findings": "No retry patterns detected (all RUN_NO = 1).", "chart": None, "severity": 1}
            
        retry_totals = df1_temp[df1_temp['RUN_NO'] > 1].groupby('ORG_NM').size()
        successful_retries = df1_temp[(df1_temp['Outcome'] == 'Succeeded (After Retry)')].groupby('ORG_NM').size()
        
        total_retries = retry_totals.sum()
        total_success_retries = successful_retries.sum()
        
        retry_lift = 0
        if total_retries > 0:
            retry_lift = (total_success_retries / total_retries) * 100
            
        findings = f"Detected **{total_retries} total retries** across the pipeline. "
        findings += f"The overall **Retry Lift rate is {retry_lift:.1f}%** (failed records successfully recovered). "
        
        if retry_lift < 10 and total_retries > 10:
            findings += "Retrying is currently ineffective for this dataset and is adding unnecessary pipeline latency. "
            severity = 3
        elif retry_lift > 40:
            findings += "Automated recovery is highly effective, successfully resolving transient data issues. "
            severity = 2
            
        findings += f"\n\n→ **Recommended action:** { 'Shift focus toward fixing root causes in upstream source systems' if retry_lift < 20 else 'Expand auto-retry frequency for high-recovery stages' } to optimize the overall processing efficiency."
            
        # Build stacked bar chart
        # Filter to top 10 most active orgs to keep chart clean
        top_orgs = df1_temp['ORG_NM'].value_counts().head(10).index
        plot_df = retry_stats[retry_stats['ORG_NM'].isin(top_orgs)]
        
        fig = px.bar(
            plot_df, 
            x='ORG_NM', 
            y='Count', 
            color='Outcome', 
            title="Pipeline Outcomes (First-Try vs Retries vs Failures)",
            color_discrete_map={
                'Succeeded (First Try)': '#2ca02c',     # green
                'Succeeded (After Retry)': '#ff7f0e', # orange
                'Failed (Permanent)': '#d62728'       # red
            }
        )
        fig.update_layout(barmode='stack', xaxis_title="Organization", yaxis_title="Number of Files (ROs)")
        
        return {
            "findings": findings,
            "chart": fig,
            "severity": severity
        }
