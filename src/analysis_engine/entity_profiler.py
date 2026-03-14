import pandas as pd
import duckdb
import plotly.graph_objects as go
from .base_module import BaseAnalysisModule

class EntityProfiler(BaseAnalysisModule):
    """
    Module 1 — Entity Profiling ("character analysis")
    What it does: Builds a complete profile for every unique entity (ORG_NM, CNT_STATE, SRC_SYS).
    What it computes: Failure rate, average retry count, most common failure reason, health score, rankings.
    Output: Ranked table + a bar chart of top 10 worst orgs.
    """

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        severity = 1
        
        # Use DuckDB to natively aggregate the Pandas DataFrame
        query = '''
        SELECT 
            ORG_NM,
            COUNT(*) as total_ros,
            SUM(CASE WHEN IS_FAILED = 1 THEN 1 ELSE 0 END) as failed_ros,
            AVG(RUN_NO) as avg_retries,
            MODE(FAILURE_STATUS) as most_common_failure
        FROM df1
        WHERE ORG_NM IS NOT NULL
        GROUP BY ORG_NM
        '''
        
        try:
            result_df = duckdb.query(query).to_df()
        except Exception as e:
            return {"findings": f"DuckDB Query failed: {str(e)}", "chart": None, "severity": 1}
            
        if len(result_df) == 0:
            return {"findings": "No organization data found to profile.", "chart": None, "severity": 1}
            
        # Calculate failure rate
        result_df['failure_rate'] = result_df['failed_ros'] / result_df['total_ros']
        
        # Health score: 100 is perfect. We subtract points for high failure rate and high retries.
        result_df['health_score'] = (1 - result_df['failure_rate']) * 100 - (result_df['avg_retries'] - 1) * 10
        
        # Rank by health score (lower is worse)
        result_df = result_df.sort_values(by='health_score', ascending=True)
        
        worst_orgs = result_df.head(10)
        
        # Generate Plotly Bar Chart
        fig = go.Figure(data=[
            go.Bar(
                x=worst_orgs['ORG_NM'],
                y=worst_orgs['failure_rate'],
                text=(worst_orgs['failure_rate'] * 100).round(1).astype(str) + '%',
                textposition='auto',
                marker_color='indianred'
            )
        ])
        
        fig.update_layout(
            title="Top Organizations by Failure Rate",
            xaxis_title="Organization",
            yaxis_title="Failure Rate",
            yaxis_tickformat='.0%'
        )
        
        # Generate insight text
        top_w_name = worst_orgs.iloc[0]['ORG_NM']
        top_w_fail_rt = worst_orgs.iloc[0]['failure_rate']
        
        findings = f"Profiled {len(result_df)} organizations. "
        findings += f"The lowest health score belongs to **{top_w_name}** with a {top_w_fail_rt:.1%} failure rate. "
        
        common_fail = worst_orgs.iloc[0]['most_common_failure']
        if pd.notna(common_fail):
            findings += f"Their most frequent failure reason is **'{common_fail}'**. "
            
        findings += f"\n\n→ **Recommended action:** Schedule a technical sync with the {top_w_name} interface team to resolve persistent '{common_fail}' errors in their source extractor."
            
        # Assess severity
        if top_w_fail_rt > 0.5:
            severity = 5
        elif top_w_fail_rt > 0.3:
            severity = 4
        elif top_w_fail_rt > 0.1:
            severity = 3
        elif top_w_fail_rt > 0.05:
            severity = 2
            
        return {
            "findings": findings,
            "chart": fig,
            "severity": severity
        }
