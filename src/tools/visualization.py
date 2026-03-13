import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import io

class VisualizationTool:
    """Generates Plotly JSON to be picked up and rendered by the Streamlit frontend."""
    
    def __init__(self):
        self.generated_charts = []  # Store for Streamlit to access later

    def generate_chart(self, chart_type: str, data_json: str, title: str, x_axis: str, y_axis: str, color_axis: str = None) -> str:
        """
        Creates a chart and saves its JSON representation to memory.
        The Streamlit UI will render any chart found in self.generated_charts.
        
        Args:
            chart_type: "bar", "line", "scatter", "pie", or "heatmap"
            data_json: A JSON string list of dictionaries containing the data. Example: '[{"A": 1, "B": 2}]'
            title: Title of the chart
            x_axis: Column name for X-axis
            y_axis: Column name for Y-axis
            color_axis: Optional column name for color grouping
            
        Returns:
            A string confirmation that the chart was generated for the Streamlit UI.
        """
        try:
            data = json.loads(data_json)
            df = pd.DataFrame(data)
            
            if df.empty:
                return "Error: Empty dataset provided for visualization."
                
            fig = None
            if chart_type.lower() == "bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_axis, title=title)
            elif chart_type.lower() == "line":
                fig = px.line(df, x=x_axis, y=y_axis, color=color_axis, title=title)
            elif chart_type.lower() == "scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_axis, title=title)
            elif chart_type.lower() == "pie":
                fig = px.pie(df, names=x_axis, values=y_axis, title=title)
            else:
                return f"Error: Unsupported chart type '{chart_type}'. Use bar, line, scatter, or pie."
                
            # Store the figure JSON so streamlit can grab it from outside
            self.generated_charts.append(fig.to_json())
            
            return f"Chart '{title}' of type '{chart_type}' generated successfully. The user can now see it in the UI."
            
        except Exception as e:
            return f"Chart generation failed: {str(e)}"
