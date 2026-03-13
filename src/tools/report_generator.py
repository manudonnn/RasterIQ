import json
from jinja2 import Template

class ReportGeneratorTool:
    """Uses Jinja2 to render an HTML report of pipeline health."""
    
    def __init__(self):
        self.generated_reports = []  # Store for Streamlit to render
        self.template_str = """
        <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
            <h2 style="color: #2c3e50;">RosterIQ Pipeline Health Report</h2>
            <h3>Target Entity: {{ target }}</h3>
            
            <div style="margin: 20px 0;">
                <h4 style="color: #e74c3c;">Critical Flags: {{ flags | length }}</h4>
                <ul>
                {% for flag in flags %}
                    <li>{{ flag }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div style="margin: 20px 0;">
                <h4 style="color: #2980b9;">Metrics Summary</h4>
                <p>{{ summary_text }}</p>
            </div>
            
            <hr>
            <p style="font-size: 12px; color: #7f8c8d;">Generated automatically by RosterIQ Agent</p>
        </div>
        """

    def generate_report(self, target: str, flags_json: str, summary_text: str) -> str:
        """
        Generates an HTML report from JSON data and saves it to memory.
        
        Args:
            target: The state, org, or client being reported on (e.g. 'Kansas (KS)')
            flags_json: A JSON list of critical issues. Example: '["3 ROs stuck in DART_GEN", "SCS_PERCENT down 5%"]'
            summary_text: A paragraph summarizing the overall health and recommendations.
        """
        try:
            flags = json.loads(flags_json)
            template = Template(self.template_str)
            
            html_content = template.render(
                target=target,
                flags=flags,
                summary_text=summary_text
            )
            
            self.generated_reports.append(html_content)
            
            return f"Health Report for '{target}' successfully generated. The user can view it in the UI."
            
        except Exception as e:
            return f"Report generation failed: {str(e)}"
