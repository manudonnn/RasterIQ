import duckdb
import pandas as pd
import json

class DuckDBTool:
    """Provides a SQL interface over the CSV data for the agent."""
    
    def __init__(self, 
                 csv1_path="data/roster_processing_details.csv", 
                 csv2_path="data/aggregated_operational_metrics.csv"):
        self.conn = duckdb.connect(database=':memory:')
        
        # Load CSVs into DuckDB tables if they exist
        try:
            self.conn.execute(f"CREATE TABLE roster_processing_details AS SELECT * FROM read_csv_auto('{csv1_path}')")
        except Exception as e:
            print(f"Warning: Could not load {csv1_path}: {e}")
            
        try:
            self.conn.execute(f"CREATE TABLE aggregated_operational_metrics AS SELECT * FROM read_csv_auto('{csv2_path}')")
        except Exception as e:
            print(f"Warning: Could not load {csv2_path}: {e}")

    def run_sql(self, query: str) -> str:
        """Executes a SQL query against the loaded data and returns results as a string."""
        try:
            # Check for modifying queries to prevent writes
            if any(kw in query.upper() for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
                return "Error: Only SELECT queries are allowed."
                
            result = self.conn.execute(query).fetchdf()
            if result.empty:
                return "Query executed successfully, but returned no results."
                
            # Limit rows returned to LLM to prevent context overflow
            if len(result) > 50:
                truncated = len(result) - 50
                res_str = result.head(50).to_string(index=False)
                return f"{res_str}\n\n... (Output truncated, {truncated} more rows)"
                
            return result.to_string(index=False)
            
        except Exception as e:
            return f"SQL Error: {str(e)}"
