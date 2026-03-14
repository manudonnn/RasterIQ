from typing import Dict, Any

class ReportBuilder:
    """
    Assembles final visual and text outputs into a formatted report (PDF, HTML, 
    or just Streamlit compatible output blocks).
    """

    def __init__(self):
        pass

    def build_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Takes raw dictionaries from the modules and compiles them into a unified
        executive summary string. 
        """
        # TODO: send this aggregate data to an LLM chain to generate the true summary
        lines = ["## Autonomous Analysis Engine Output Summary", ""]
        
        for mod_name, res in analysis_results.items():
            lines.append(f"### {mod_name.replace('_', ' ').title()}")
            lines.append(f"**Severity:** {res.get('severity', 'N/A')}/5")
            lines.append(f"**Findings:** {res.get('findings', 'None')}")
            lines.append("---")
            
        return "\n".join(lines)
