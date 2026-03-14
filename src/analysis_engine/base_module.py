import pandas as pd
from typing import Optional
from .scope import AnalysisScope

class BaseAnalysisModule:
    """
    Abstract class that all analysis modules inherit from.
    """
    
    def _apply_scope(self, df: pd.DataFrame, scope: AnalysisScope) -> pd.DataFrame:
        if scope is None:
            return df
            
        filtered_df = df.copy()
        
        # We check for columns to ensure we don't break if a df doesn't have the col
        if scope.states and 'CNT_STATE' in filtered_df.columns:
            if "ALL" not in [s.upper() for s in scope.states]:
                filtered_df = filtered_df[filtered_df['CNT_STATE'].isin(scope.states)]
                
        if scope.orgs and 'ORG_NM' in filtered_df.columns:
            if "ALL" not in [o.upper() for o in scope.orgs]:
                filtered_df = filtered_df[filtered_df['ORG_NM'].isin(scope.orgs)]
                
        if scope.lobs and 'LOB' in filtered_df.columns:
            if "ALL" not in [l.upper() for l in scope.lobs]:
                filtered_df = filtered_df[filtered_df['LOB'].isin(scope.lobs)]
                
        # Limit rows as specified in scope for specific large tables if desired,
        # but generally we want to summarize the entire scope matching the criteria.
        
        return filtered_df
    
    def run(self, df1: pd.DataFrame, df2: pd.DataFrame, scope: Optional[AnalysisScope] = None) -> dict:
        """
        Filters data by scope if provided, then executes the analysis.
        """
        filtered_df1 = self._apply_scope(df1, scope)
        filtered_df2 = self._apply_scope(df2, scope)
        
        return self.analyze(filtered_df1, filtered_df2)

    def analyze(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        """
        Internal execution logic for each specific module.
        """
        raise NotImplementedError("Each module must implement the analyze() method.")
