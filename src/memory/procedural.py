import os
import yaml
from glob import glob

class ProceduralMemory:
    """Loads operational procedures (diagnostic workflows) from YAML files."""
    
    def __init__(self, procedures_dir="src/knowledge/procedures"):
        self.procedures_dir = procedures_dir
        self.procedures = {}
        self.load_all()

    def load_all(self):
        """Loads all YAML files dynamically from the directory."""
        if not os.path.exists(self.procedures_dir):
            return

        for filepath in glob(os.path.join(self.procedures_dir, "*.yaml")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    proc_data = yaml.safe_load(f)
                    if proc_data and "name" in proc_data:
                        self.procedures[proc_data["name"]] = proc_data
            except Exception as e:
                print(f"Failed to load procedure {filepath}: {e}")

    def list_procedures(self) -> str:
        """Returns a formatted list of available procedures to the agent."""
        if not self.procedures:
            return "No operational procedures available."
            
        lines = []
        for name, data in self.procedures.items():
            lines.append(f"- {name}: {data.get('description', 'No description.')}")
        return "Available Procedural Memory Workflows:\n" + "\n".join(lines)

    def get_procedure(self, name: str) -> str:
        """Retrieves exactly how to execute a specific stored procedure."""
        if name not in self.procedures:
            return f"Procedure '{name}' not found. Available: {', '.join(self.procedures.keys())}"
            
        data = self.procedures[name]
        return (
            f"Procedure Execution Guide for: {name}\n"
            f"Description: {data.get('description')}\n"
            f"Parameters: {data.get('parameters')}\n"
            f"Query/Action: {data.get('sql_query')}\n"
            f"Instruction: {data.get('instruction')}\n"
        )
